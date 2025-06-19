import open3d as o3d
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
        rgb_img = x[..., :3]
        depth_img = x[..., 3:4]
        seg_img = x[..., 4:5]

        # unproject pixels to 3D points (negative because of meshgrid indexing)
        Z = depth_img
        X = self.u_grid * Z
        Y = self.v_grid * Z
        points = torch.cat((X, Y, Z), dim=-1)

        # mask points based on depth and segmentation
        mask = (Z < 3.0).squeeze(-1)
        seg_mask = (seg_img != 15).squeeze(-1)
        points = points[mask & seg_mask]
        rgb_img = rgb_img[mask & seg_mask] / 255.0

        # transform points to world coordinates
        t = (
            Transform3d(device=self.device)
            .rotate_axis_angle(33)
            .translate(0, 1.5, -2.5)
        )
        world_points = t.transform_points(points)

        # downsample
        world_points, indices = sample_farthest_points(world_points.unsqueeze(0), K=512)
        rgb_img = rgb_img[indices]

        breakpoint()
        return torch.cat((world_points, rgb_img), dim=-1)
