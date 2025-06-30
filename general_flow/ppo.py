import types

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from general_flow.cotracker_wrappers import batched_forward_window, masked_forward
from general_flow.rollout_storage import OpticalFlowRolloutStorage


class OpticalFlowPPO(PPO):
    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        self.transition = OpticalFlowRolloutStorage.Transition()
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = OpticalFlowRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def load_cotracker(self):
        # load the keypoint tracking data and segmentation masked_forward
        # these are hard coded for now but should come from flow prediction model
        self.cotracker_data = torch.load(
            "tests/cotracker_output.pt", map_location=self.device
        )
        segm_mask = np.load("tests/seg_masks.npy")
        self.segm_mask = torch.tensor(segm_mask, dtype=torch.float32, device=self.device)[None, None]

        # load the CoTracker model
        cotracker_model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(self.device)  # type: ignore
        cotracker_model.forward = types.MethodType(masked_forward, cotracker_model)
        cotracker_model.model.forward_window = types.MethodType(batched_forward_window, cotracker_model.model)
        self.cotracker_model = cotracker_model

    def act(self, obs, critic_obs, camera_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        self.transition.camera_observations = camera_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        video = self.storage.camera_observations
        video = video.permute(1, 0, 4, 2, 3).float()

        # Initialize online processing
        self.cotracker_model(video_chunk=video, is_first_step=True, grid_size=50, segm_mask=self.segm_mask)

        # Process the video
        for ind in range(0, video.shape[1] - self.cotracker_model.step, self.cotracker_model.step):
            pred_tracks, pred_visibility = self.cotracker_model(
                video_chunk=video[:, ind : ind + self.cotracker_model.step * 2]
            )

        gt = self.cotracker_data["pred_tracks"][:, :self.storage.num_transitions_per_env]

        # TODO: normalize gt and pred tracks by heigth and width (units are pixels)

        rewards = torch.norm(pred_tracks - gt, dim=-1).mean(dim=-1, keepdim=True)
        self.storage.rewards = rewards.permute(1, 0, 2)

        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )
