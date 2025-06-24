from cotracker.predictor import get_points_on_a_grid
import torch
import torch.nn.functional as F


@torch.no_grad()
def masked_forward(
    self,
    video_chunk,
    is_first_step: bool = False,
    queries: torch.Tensor | None = None,
    segm_mask: torch.Tensor | None = None,
    grid_size: int = 5,
    grid_query_frame: int = 0,
    add_support_grid=False,
):
    B, T, C, H, W = video_chunk.shape
    # Initialize online video processing and save queried points
    # This needs to be done before processing *each new video*
    if is_first_step:
        self.model.init_video_online_processing()
        if queries is not None:
            B, N, D = queries.shape
            self.N = N
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
            if add_support_grid:
                grid_pts = get_points_on_a_grid(
                    self.support_grid_size,
                    self.interp_shape,
                    device=video_chunk.device,
                )
                grid_pts = torch.cat(
                    [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
                )
                queries = torch.cat([queries, grid_pts], dim=1)
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(
                grid_size, self.interp_shape, device=video_chunk.device
            )
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(self.interp_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]
            self.N = grid_size**2
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            )

        self.queries = queries
        return (None, None)

    video_chunk = video_chunk.reshape(B * T, C, H, W)
    video_chunk = F.interpolate(
        video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
    )
    video_chunk = video_chunk.reshape(
        B, T, 3, self.interp_shape[0], self.interp_shape[1]
    )
    if self.v2:
        tracks, visibilities, __ = self.model(
            video=video_chunk, queries=self.queries, iters=6, is_online=True
        )
    else:
        tracks, visibilities, confidence, __ = self.model(
            video=video_chunk, queries=self.queries, iters=6, is_online=True
        )
    if add_support_grid:
        tracks = tracks[:, :, : self.N]
        visibilities = visibilities[:, :, : self.N]
        if not self.v2:
            confidence = confidence[:, :, : self.N]

    if not self.v2:
        visibilities = visibilities * confidence
    thr = 0.6
    return (
        tracks
        * tracks.new_tensor(
            [
                (W - 1) / (self.interp_shape[1] - 1),
                (H - 1) / (self.interp_shape[0] - 1),
            ]
        ),
        visibilities > thr,
    )
