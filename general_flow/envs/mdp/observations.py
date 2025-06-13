from __future__ import annotations

import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def camera_rgb(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["tiled_camera"].data.output["rgb"].flatten(start_dim=1)
