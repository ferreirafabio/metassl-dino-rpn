import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


data_path = "../../datasets/CIFAR10"
cifar = datasets.CIFAR10(data_path)
image = cifar.__getitem__(1111)[0]

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

identity = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float)
theta = torch.rand(1, 2, 3)
print(theta)

x = toTensor(image).unsqueeze(0)
loss1 = nn.MSELoss()(theta, identity)
print("loss1: ", loss1.item())
print()
loss2 = nn.MSELoss()(theta.expand(8, 2, 3), identity)
print("loss2: ", loss2.item())
print()
print("theta:", theta.shape)
print()
print("x:", x.shape)
print()
grid = F.affine_grid(theta, x.size(), align_corners=False)
print("grid:", grid.shape)
print()

id_grid = F.affine_grid(identity, x.size(), align_corners=True)
print(id_grid.shape)
loss2 = nn.MSELoss()(grid, id_grid)
print("loss2: ", loss1.item())
print()
y = F.grid_sample(x, grid, align_corners=False)


# f = plt.figure()
# f.add_subplot(1, 2, 1)
# plt.imshow(image)
# f.add_subplot(1, 2, 2)
# plt.imshow(toPIL(y.squeeze()))
# plt.show()
