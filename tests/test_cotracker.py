import time
import types

import imageio.v3 as iio
import numpy as np
import torch
from cotracker.utils.visualizer import Visualizer

from general_flow.masked_online_predictor import masked_forward

RUN_OFFLINE = False
RUN_ONLINE = True


def get_performance_metrics(start_time):
    # Get the maximum GPU memory allocated
    max_memory_bytes = torch.cuda.max_memory_allocated()
    max_memory_gb = max_memory_bytes / (1024**3)
    print(f"\nMaximum GPU Memory Usage: {max_memory_gb:.2f} GB")

    # Get the total time taken for the operation
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time Taken: {total_time:.2f} seconds")


frames = iio.imread("tests/video.mp4", plugin="FFMPEG")  # plugin="pyav"
device = "cuda"
grid_size = 80
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

segm_mask = np.load("tests/seg_masks.npy")
segm_mask = torch.tensor(segm_mask, dtype=torch.float32, device=device)[None, None]

# Load the CoTracker models
if RUN_OFFLINE:
    offline_cotracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_offline"
    ).to(device)  # type: ignore
if RUN_ONLINE:
    online_cotracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_online"
    ).to(device)  # type: ignore
    # need this because there's no masking for the online model
    online_cotracker.forward = types.MethodType(masked_forward, online_cotracker)

# Run Offline CoTracker:
if RUN_OFFLINE:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    pred_tracks, pred_visibility = offline_cotracker(
        video, grid_size=grid_size, segm_mask=segm_mask
    )
    get_performance_metrics(start)

# Run Online CoTracker:
if RUN_ONLINE:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    online_cotracker(
        video_chunk=video, is_first_step=True, grid_size=grid_size, segm_mask=segm_mask
    )
    for ind in range(0, video.shape[1] - online_cotracker.step, online_cotracker.step):
        pred_tracks, pred_visibility = online_cotracker(
            video_chunk=video[:, ind : ind + online_cotracker.step * 2]
        )
    get_performance_metrics(start)

# save data
data = {"pred_tracks": pred_tracks, "pred_visibility": pred_visibility}
torch.save(data, "tests/cotracker_output.pt")

# Visualize the results
vis = Visualizer(save_dir="./saved_videos", pad_value=0, linewidth=3)
vis.visualize(video, pred_tracks, pred_visibility)
