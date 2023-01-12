import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()


data_path = "../../datasets/CIFAR10"
cifar = datasets.CIFAR10(data_path)
toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

# theta = torch.tensor([[[-1.5, 0.25, 0.], [0.5, 1.5, 0.]]],)
theta = torch.randn(1, 2, 3)

x = cifar.__getitem__(torch.randint(50000, (1,)).item())[0]
x = toTensor(x).unsqueeze(0)
grid = F.affine_grid(theta, x.size(), align_corners=True)
a = F.grid_sample(x, grid, align_corners=True)

theta_tanh = torch.tanh(theta)
grid = F.affine_grid(theta_tanh, x.size(), align_corners=True)
b = F.grid_sample(x, grid, align_corners=True)

theta_norm = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True)
grid = F.affine_grid(theta_norm, x.size(), align_corners=True)
c = F.grid_sample(x, grid, align_corners=True)

theta_all = theta_tanh / torch.linalg.norm(theta_tanh, ord=1, dim=2, keepdim=True)
grid = F.affine_grid(theta_all, x.size(), align_corners=True)
d = F.grid_sample(x, grid, align_corners=True)

images = [a, b, c, d]
images = [toPIL(img.squeeze()) for img in images]
plot(images)
plt.show()
