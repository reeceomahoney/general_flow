import types

import numpy as np
import torch
import torch.nn.functional as F
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
        cotracker_data = torch.load(
            "tests/cotracker_output.pt", map_location=self.device
        )
        self.cotracker_data = self._upsample_timesteps(cotracker_data["pred_tracks"], upsample_factor=5)
        segm_mask = np.load("tests/seg_masks.npy")
        self.segm_mask = torch.tensor(segm_mask, dtype=torch.float32, device=self.device)[None, None]

        # load the CoTracker model
        cotracker_model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(self.device)  # type: ignore
        cotracker_model.forward = types.MethodType(masked_forward, cotracker_model)
        cotracker_model.model.forward_window = types.MethodType(batched_forward_window, cotracker_model.model)
        self.cotracker_model = cotracker_model

    def _upsample_timesteps(self, x, upsample_factor):
        # we need this to get the data from 10 fps to 50 fps
        b, t, c1, c2 = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(b * c1 * c2, 1, t)
        x_upsampled = F.interpolate(x_reshaped, scale_factor=upsample_factor, mode='linear', align_corners=False)
        new_l = x_upsampled.shape[2]
        return x_upsampled.reshape(b, c1, c2, new_l).permute(0, 3, 1, 2)

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

    def compute_returns(self, last_critic_obs, timesteps):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        video = self.storage.camera_observations
        video = video.permute(1, 0, 4, 2, 3).float()

        # using timesteps[0] is a hack as these are all the same for now, will need to fix when we add terminations
        curr_t = timesteps[0]

        # split the video into chunks based on dones
        video_chunks = []
        if self.storage.dones.any() and curr_t != 0:
            split_idx = self.storage.dones.nonzero(as_tuple=True)[0]
            split_idx = int(split_idx[0] + 1)
            ep1, ep2 = video[:, :split_idx], video[:, split_idx:]
            video_chunks.append(ep1)
            video_chunks.append(ep2)
        else:
            video_chunks.append(video)
        
        # TODO: we need to recompute the segm_mask for each episode

        # run the CoTracker model on the video chunks
        tracks_list = []
        for v in video_chunks:
            self.cotracker_model(video_chunk=v, is_first_step=True, grid_size=50, segm_mask=self.segm_mask)
            if v.shape[1] <= self.cotracker_model.step:
                pred_tracks, pred_visibility = self.cotracker_model(
                    video_chunk=v
                )
            else:
                for ind in range(0, v.shape[1] - self.cotracker_model.step, self.cotracker_model.step):
                    pred_tracks, pred_visibility = self.cotracker_model(
                        video_chunk=v[:, ind : ind + self.cotracker_model.step * 2]
                    )
            tracks_list.append(pred_tracks)

        tracks = torch.cat(tracks_list, dim=1)

        gt_tracks = []
        delta = self.storage.num_transitions_per_env
        if self.storage.dones.any() and curr_t != 0:
            gt_tracks.append(self.cotracker_data[:, -split_idx:])
            gt_tracks.append(self.cotracker_data[:, :curr_t])
        elif curr_t == 0:
            gt_tracks.append(self.cotracker_data[:, -delta:])
        else:
            gt_tracks.append(self.cotracker_data[:, curr_t - delta : curr_t])

        gt = torch.cat(gt_tracks, dim=1)
        scale = torch.tensor(
            [video.shape[3], video.shape[4]], device=self.device
        ).view(1, 1, 1, 2)

        rewards = torch.norm((tracks - gt) / scale, dim=-1).mean(dim=-1, keepdim=True)
        self.storage.rewards = rewards.permute(1, 0, 2)

        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

        return rewards
