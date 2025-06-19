import torch
import torch.nn as nn
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import Transform3d
from termcolor import cprint


class PointNetEncoder(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        image_shape: tuple,
        out_channels: int = 64,
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        device: str = "cuda",
    ):
        """_summary_

        Args:
            image_shape (tuple): shape of the input image (H, W, C)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), "cyan")
        cprint("pointnet use_final_norm: {}".format(final_norm), "cyan")

        self.mlp = nn.Sequential(
            nn.Linear(6, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        # camera intrinsics
        focal_length = 50
        width = image_shape[0]
        fx = width * focal_length / 20.995

        # create image grid
        u = torch.arange(width, device=device)
        v = torch.arange(width, device=device)
        u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
        u_grid = u_grid.unsqueeze(-1)
        v_grid = v_grid.unsqueeze(-1)

        # negative because of meshgrid indexing
        self.u_grid = -(u_grid - width // 2) / fx
        self.v_grid = -(v_grid - width // 2) / fx

        # set device
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self._create_point_cloud(x)
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x

    def _create_point_cloud(self, x):
        # load RGB and depth images
        rgb_img = x[..., :3] / 255.0
        depth_img = x[..., 3:4]
        seg_img = x[..., 4:5]
        B = rgb_img.shape[0]

        # unproject pixels to 3D points
        Z = depth_img
        X = self.u_grid * Z
        Y = self.v_grid * Z
        points = torch.cat((X, Y, Z), dim=-1)

        # mask points based on depth and segmentation
        mask = (Z < 3.0).expand_as(points)
        seg_mask = (seg_img != 2).expand_as(points)
        points = torch.where(mask & seg_mask, points, torch.zeros_like(points))
        rgb_img = torch.where(mask & seg_mask, rgb_img, torch.zeros_like(rgb_img))

        # transform points to world coordinates
        t = (
            Transform3d(device=self.device)
            .rotate_axis_angle(33)
            .translate(0, 1.5, -2.5)
        )
        world_points = t.transform_points(points.reshape(B, -1, 3))

        # downsample
        world_points_downsample, indices = sample_farthest_points(world_points, K=512)
        indices = indices.unsqueeze(-1).expand(-1, -1, 3)
        rgb_downsample = torch.gather(rgb_img.reshape(B, -1, 3), 1, indices)

        return torch.cat((world_points_downsample, rgb_downsample), dim=-1)
