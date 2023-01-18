import torch
import torch.nn.functional as F

theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)

grid = F.affine_grid(theta, [1, 1, 16, 16], align_corners=False)
print(grid.shape)
print(grid[0, 0, 0, :])
print(grid[0, 0, -1, :])
print(grid[0, -1, 0, :])
a = torch.pow((grid[0, 0, 0, :] - grid[0, 0, -1, :]) / 2, 2).sum().sqrt()
print(a)
b = torch.pow((grid[0, 0, 0, :] - grid[0, -1, 0, :]) / 2, 2).sum().sqrt()
print(b)
print(a * b)
