import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

toP = transforms.ToPILImage()
toT = transforms.ToTensor()

data_path = "../../datasets/CIFAR10"
cifar = datasets.CIFAR10(data_path)

image, _ = cifar.__getitem__(0)

theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
align = True
angle = torch.tensor(torch.pi/4)
scale = 1
scale2 = 1
ident = torch.tensor([[[1, 0, 0],
                       [0, 1, 0]]], dtype=torch.float)
rand = torch.rand(1, 2, 3)
thetangle = torch.tensor([[
    [scale*torch.cos(angle), -torch.sin(angle), 0],
    [torch.sin(angle), scale*torch.cos(angle), 0]
]])

theta = ident
img = toT(image).unsqueeze(0)
print(theta)

grid1 = F.affine_grid(theta, size=img.shape, align_corners=align)
print(grid1[0, 0, 0, :])
print(grid1[0, 0, -1, :])
print(grid1[0, -1, 0, :])
print(grid1[0, -1, -1, :])
out1 = F.grid_sample(img, grid1, align_corners=align)

theta2 = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True)
print(f"{theta2=}")
grid2 = F.affine_grid(theta2, size=img.shape, align_corners=False)
out2 = F.grid_sample(img, grid2, align_corners=False)
print(grid2[0, 0, 0, :])
print(grid2[0, 0, -1, :])
print(grid2[0, -1, 0, :])
print(grid2[0, -1, -1, :])
print(img.sum() == out2.sum())
theta3 = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)
grid3 = F.affine_grid(theta3, size=img.shape, align_corners=align)
out3 = F.grid_sample(img, grid3, align_corners=align)

print(theta.abs().sum((1, 2), keepdim=True))
theta4 = theta / torch.linalg.norm(theta, ord=2, dim=2, keepdim=True)  # .clamp(min=1)
print(f"{theta4.tolist()=}")
# theta4 = theta / theta.abs().mean((1, 2), keepdim=True)
# print(f"{theta4.tolist()=}")
grid4 = F.affine_grid(theta4, size=img.shape, align_corners=align)
out4 = F.grid_sample(img, grid4, align_corners=align)

plt.subplot(1, 5, 1)
plt.imshow(image)
plt.subplot(1, 5, 2)
plt.imshow(toP(out1.squeeze()))
plt.subplot(1, 5, 3)
plt.imshow(toP(out2.squeeze()))
plt.subplot(1, 5, 4)
plt.imshow(toP(out3.squeeze()))
plt.subplot(1, 5, 5)
plt.imshow(toP(out3.squeeze()))
plt.show()
