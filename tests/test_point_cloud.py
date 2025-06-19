import torch
import open3d as o3d
from pytorch3d.transforms import Transform3d
from pytorch3d.ops import sample_farthest_points

# load RGB and depth images
rgb_img = torch.load("tests/rgb.pt")
depth_img = torch.load("tests/depth.pt")
seg_img = torch.load("tests/seg.pt")
device = "cuda"

# camera intrinsics
focal_length = 50
width = 84
fx = (width * focal_length) / 20.995

# create image grid
u = torch.arange(width, device=device)
v = torch.arange(width, device=device)
u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
u_grid = u_grid.unsqueeze(-1)
v_grid = v_grid.unsqueeze(-1)

# unproject pixels to 3D points (negative because of meshgrid indexing)
Z = depth_img
X = -(u_grid - width // 2) * Z / fx
Y = -(v_grid - width // 2) * Z / fx
points = torch.cat((X, Y, Z), dim=-1)

# truncate points based on depth
mask = (Z < 3.0).squeeze(-1)
seg_mask = (seg_img != 15).squeeze(-1)
points = points[mask & seg_mask]
rgb_img = rgb_img[mask & seg_mask]

# transform points to world coordinates
t = Transform3d(device=device).rotate_axis_angle(33).translate(0, 1.5, -2.5)
world_points = t.transform_points(points)

world_points = sample_farthest_points(world_points.unsqueeze(0), K=512)[0]

# visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(world_points[0].cpu().numpy())
pcd.colors = o3d.utility.Vector3dVector(rgb_img.cpu().numpy() / 255.0)
o3d.visualization.draw_geometries([pcd])  # type: ignore
