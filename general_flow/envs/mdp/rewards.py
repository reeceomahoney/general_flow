import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def keypoint_tracking(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the agent for tracking the keypoints."""
    # cotracker_model = env.reward_manager.cotracker_model
    # cotracker_data = env.reward_manager.cotracker_data
    # step = env.episode_length_buf
    # is_first_step = step == 1
    #
    # if is_first_step:
    #     keypoint_tracking.frames = []
    # frames = keypoint_tracking.frames
    # rgb = env.scene["tiled_camera"].data.output["rgb"]
    # frames.append(rgb)
    #
    # if step - 1 % cotracker_model.step == 0 and not is_first_step:
    #     video_chunk = torch.stack(frames[-cotracker_model.step * 2 :])
    #     video_chunk = video_chunk.permute(0, 2, 1, 3, 4)
    #
    #     if is_first_step:
    #         seg = env.scene["tiled_camera"].data.output["semantic_segmentation"]
    #         segm_mask = (seg == 2).squeeze()
    #     else:
    #         segm_mask = None
    #
    #     pred_tracks, pred_visibility = cotracker_model(
    #         video_chunk=video_chunk,
    #         is_first_step=(step == 0),
    #         grid_size=50,
    #         segm_mask=segm_mask,
    #     )
    # elif step == env.max_episode_length:
    #     video_chunk = torch.stack(
    #         frames[-(step % cotracker_model.step) - cotracker_model.step - 1 :]
    #     )
    #     video_chunk = video_chunk.permute(0, 2, 1, 3, 4)
    #     pred_tracks, pred_visibility = cotracker_model(
    #         video_chunk=video_chunk,
    #         is_first_step=False,
    #         grid_size=50,
    #     )
    #
    # err = torch.norm(pred_tracks - cotracker_data["pred_tracks"], dim=-1)
    # return err

    return torch.zeros(env.num_envs, device=env.device)
