from cotracker.models.core.cotracker.cotracker3_online import posenc
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
            ).repeat(B, 1, 1)

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


def batched_forward_window(
    self,
    fmaps_pyramid,
    coords,
    track_feat_support_pyramid,
    vis=None,
    conf=None,
    attention_mask=None,
    iters=4,
    add_space_attn=False,
):
    B, S, *_ = fmaps_pyramid[0].shape
    N = coords.shape[2]
    r = 2 * self.corr_radius + 1

    coord_preds, vis_preds, conf_preds = [], [], []
    for it in range(iters):
        coords = coords.detach()  # B T N 2
        coords_init = coords.reshape(B * S, N, 2)
        corr_embs = []
        corr_feats = []
        for i in range(self.corr_levels):
            corr_feat = self.get_correlation_feat(fmaps_pyramid[i], coords_init / 2**i)
            track_feat_support = (
                track_feat_support_pyramid[i]
                .view(B, 1, r, r, N, self.latent_dim)
                .squeeze(1)
                .permute(0, 3, 1, 2, 4)
            )
            corr_volume = torch.einsum(
                "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
            )
            corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r * r))

            corr_embs.append(corr_emb)

        corr_embs = torch.cat(corr_embs, dim=-1)
        corr_embs = corr_embs.view(B, S, N, corr_embs.shape[-1])

        transformer_input = [vis, conf, corr_embs]

        rel_coords_forward = coords[:, :-1] - coords[:, 1:]
        rel_coords_backward = coords[:, 1:] - coords[:, :-1]

        rel_coords_forward = torch.nn.functional.pad(
            rel_coords_forward, (0, 0, 0, 0, 0, 1)
        )
        rel_coords_backward = torch.nn.functional.pad(
            rel_coords_backward, (0, 0, 0, 0, 1, 0)
        )

        scale = (
            torch.tensor(
                [self.model_resolution[1], self.model_resolution[0]],
                device=coords.device,
            )
            / self.stride
        )
        rel_coords_forward = rel_coords_forward / scale
        rel_coords_backward = rel_coords_backward / scale

        rel_pos_emb_input = posenc(
            torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
            min_deg=0,
            max_deg=10,
        )  # batch, num_points, num_frames, 84
        transformer_input.append(rel_pos_emb_input)

        x = (
            torch.cat(transformer_input, dim=-1)
            .permute(0, 2, 1, 3)
            .reshape(B * N, S, -1)
        )

        x = x + self.interpolate_time_embed(x, S)
        x = x.view(B, N, S, -1)  # (B N) T D -> B N T D

        delta = self.updateformer(x, add_space_attn=add_space_attn)

        delta_coords = delta[..., :2].permute(0, 2, 1, 3)
        delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
        delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

        vis = vis + delta_vis
        conf = conf + delta_conf

        coords = coords + delta_coords
        coord_preds.append(coords[..., :2] * float(self.stride))

        vis_preds.append(vis[..., 0])
        conf_preds.append(conf[..., 0])
    return coord_preds, vis_preds, conf_preds
