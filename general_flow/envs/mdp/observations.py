from __future__ import annotations

import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def camera_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["tiled_camera"].data.output["rgb"]


def camera_segm_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    segm = env.scene["tiled_camera"].data.output["semantic_segmentation"]
    return segm == 3


def camera_rgbd(env: ManagerBasedRLEnv) -> torch.Tensor:
    rgb = env.scene["tiled_camera"].data.output["rgb"]
    depth = env.scene["tiled_camera"].data.output["depth"]
    return torch.cat((rgb, depth), dim=-1).flatten(start_dim=1)


def two_cameras(env: ManagerBasedRLEnv) -> torch.Tensor:
    scene_rgb = env.scene["tiled_camera"].data.output["rgb"]
    wrist_rgb = env.scene["wrist_camera"].data.output["rgb"]
    return torch.cat((scene_rgb, wrist_rgb), dim=-1).flatten(start_dim=1)


def camera_rgb_depth_seg(env: ManagerBasedRLEnv) -> torch.Tensor:
    # rgb = env.scene["tiled_camera"].data.output["rgb"]
    depth = env.scene["tiled_camera"].data.output["depth"]
    seg = env.scene["tiled_camera"].data.output["semantic_segmentation"]
    return torch.cat((depth, seg), dim=-1).flatten(start_dim=1)
