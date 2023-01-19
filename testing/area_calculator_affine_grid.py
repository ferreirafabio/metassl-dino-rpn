import torch
import torch.nn.functional as F

theta = torch.tensor([[[0.67, 0.5, 0], [.25, 1, 0]]], dtype=torch.float)
theta = torch.rand(1, 2, 3)
theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)

print(theta)
grid = F.affine_grid(theta, [1, 1, 16, 16], align_corners=True)
print(grid[0, 0, 0, :])
print(grid[0, 0, -1, :])
print(grid[0, -1, 0, :])
print(grid[0, -1, -1, :])
a = torch.pow((grid[0, 0, 0, :] - grid[0, 0, -1, :]) / 2, 2).sum().sqrt()
print(a)
b = torch.pow((grid[0, 0, 0, :] - grid[0, -1, 0, :]) / 2, 2).sum().sqrt()
print(b)
print(a * b)
print(torch.det(theta[:,:,:2]))
