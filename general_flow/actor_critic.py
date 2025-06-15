# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ConvActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        observation_shape: dict,
        num_actions: int,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
    ):
        super().__init__()

        image_shape = observation_shape.get("image")
        state_dim = observation_shape.get("state")
        activation_fn = resolve_nn_activation(activation)

        # --- Visual Encoder (CNN) ---
        if image_shape:
            cnn_layers = []
            # Input: (C, 64, 64)
            cnn_layers.extend(
                [
                    nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
                    activation_fn,
                ]
            )
            cnn_layers.extend(
                [
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    activation_fn,
                ]
            )
            cnn_layers.extend(
                [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    activation_fn,
                ]
            )
            cnn_layers.append(nn.Flatten())
            self.visual_encoder = nn.Sequential(*cnn_layers)
            # Calculate the flattened output size from the CNN
            with torch.no_grad():
                dummy_input = torch.zeros(1, *image_shape)
                visual_feature_dim = self.visual_encoder(dummy_input).shape[1]
        else:
            self.visual_encoder = None
            visual_feature_dim = 0

        # --- State Encoder (MLP) ---
        if state_dim:
            state_encoder_layers = []
            state_encoder_layers.extend([nn.Linear(state_dim, 128), activation_fn])
            state_encoder_layers.extend([nn.Linear(128, 64), activation_fn])
            self.state_encoder = nn.Sequential(*state_encoder_layers)
            state_feature_dim = 64
        else:
            self.state_encoder = None
            state_feature_dim = 0

        # --- Fused Feature Dimension ---
        fused_feature_dim = visual_feature_dim + state_feature_dim
        if fused_feature_dim == 0:
            raise ValueError(
                "At least one observation modality (image or state) must be provided."
            )

        # --- Actor Head ---
        actor_head_layers = []
        actor_head_layers.append(nn.Linear(fused_feature_dim, actor_hidden_dims[0]))
        actor_head_layers.append(activation_fn)
        for i in range(len(actor_hidden_dims) - 1):
            actor_head_layers.append(
                nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1])
            )
            actor_head_layers.append(activation_fn)
        actor_head_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor_head = nn.Sequential(*actor_head_layers)

        # --- Critic Head ---
        critic_head_layers = []
        critic_head_layers.append(nn.Linear(fused_feature_dim, critic_hidden_dims[0]))
        critic_head_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_head_layers.append(
                nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1])
            )
            critic_head_layers.append(activation_fn)
        critic_head_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic_head = nn.Sequential(*critic_head_layers)

        print("--- Actor-Critic Multimodal Network Initialized ---")
        if self.visual_encoder:
            print(f"Visual Encoder (CNN): {self.visual_encoder}")
        if self.state_encoder:
            print(f"State Encoder (MLP): {self.state_encoder}")
        print(f"Actor Head: {self.actor_head}")
        print(f"Critic Head: {self.critic_head}")
        print("-------------------------------------------------")

        # --- Action Noise ---
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        # --- Action Distribution ---
        self.distribution = Normal(torch.zeros(num_actions), torch.ones(num_actions))
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _get_fused_features(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Processes multimodal observations through their respective encoders and fuses the features.
        """
        img_obs = observations[:, : 6 * 64 * 64]
        img_obs = img_obs.view(-1, 64, 64, 4)
        img_obs = img_obs.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        state_obs = observations[:, 6 * 64 * 64 :]
        obs_dict = {"image": img_obs, "state": state_obs}

        feature_list = []
        if self.visual_encoder:
            img_obs = obs_dict["image"]
            visual_features = self.visual_encoder(img_obs)
            feature_list.append(visual_features)

        if self.state_encoder:
            state_obs = obs_dict["state"]
            state_features = self.state_encoder(state_obs)
            feature_list.append(state_features)

        # Concatenate features from all modalities
        return torch.cat(feature_list, dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        # compute mean
        fused_features = self._get_fused_features(observations)
        mean = self.actor_head(fused_features)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        fused_features = self._get_fused_features(observations)
        actions_mean = self.actor_head(fused_features)
        return actions_mean

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        fused_features = self._get_fused_features(critic_observations)
        value = self.critic_head(fused_features)
        return value

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
