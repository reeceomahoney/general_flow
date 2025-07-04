# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--camera_video", action="store_true", default=False)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

import general_flow.envs  # noqa: F401
from general_flow.lift_cube_sm import PickAndLiftSm
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData  # isort: skip


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    print("[INFO] Parsing environment configuration.")
    env_cfg = parse_env_cfg(
        task_name="Isaac-Franka-Lift",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create isaac environment
    env = gym.make("Isaac-Franka-Lift", cfg=env_cfg)
    env.unwrapped.sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])  # type: ignore
    env.reset()

    if args_cli.camera_video:
        frames = []
        max_timesteps = 20*50

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros(
        (env.unwrapped.num_envs, 4), device=env.unwrapped.device
    )
    desired_orientation[:, 1] = 1.0
    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=0.01,
    )

    timestep = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.camera_video:
                rgb = env.unwrapped.scene["tiled_camera"].data.output["semantic_segmentation"]
                frames.append(rgb.clone().squeeze())

                timestep += 1
                if timestep == max_timesteps:
                    break

            # step environment
            dones = env.step(actions)[-2]

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
                - env.unwrapped.scene.env_origins
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[
                ..., :3
            ]

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            # reset state machine
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    if args_cli.camera_video:
        print("[INFO] Saving frames to video.")
        frames = torch.stack(frames, dim=0).cpu().numpy()
        iio.imwrite("tests/video.mp4", frames, fps=50)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
