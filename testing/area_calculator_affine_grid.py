import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import plot


if __name__ == "__main__":
    toT = transforms.ToTensor()
    toP = transforms.ToPILImage()

    data_path = "../../datasets/CIFAR10"
    cifar = datasets.CIFAR10(data_path)
    image, _ = cifar.__getitem__(0)
    images = [image]
    align = False

    x = toT(image).unsqueeze(0)
    theta = torch.tensor([[[1, 0, .5], [0, 1, .5]]], dtype=torch.float)
    grid = F.affine_grid(theta, x.shape, align_corners=align)
    out = F.grid_sample(x, grid, align_corners=align)
    images.append(toP(out.squeeze()))

    print("corners: ", grid[0, 0, 0, :].tolist(), grid[0, 0, -1, :].tolist(), grid[0, -1, 0, :].tolist(), grid[0, -1, -1, :].tolist(), sep="\n", end="\n\n")

    theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)
    print(f"{theta=}")
    a = torch.pow((grid[0, 0, 0, :] - grid[0, 0, -1, :]) / 2, 2).sum().sqrt().item()
    b = torch.pow((grid[0, 0, 0, :] - grid[0, -1, 0, :]) / 2, 2).sum().sqrt().item()
    print("length of side a: ", a)
    print("length of side b: ", b)
    print("area of new image: ", a * b, end="\n\n")
    print("area with det.: ", torch.det(theta[:, :, :2]).abs().item(), end="\n\n")

    print((theta[:, :, 2].abs() / 2))
    area_of_txy = (1 - (theta[:, :, 2].abs() / 2)).prod().pow(2).item()
    print(f"{area_of_txy=}")

    grid = F.affine_grid(theta, x.shape, align_corners=align)
    out = F.grid_sample(x, grid, align_corners=align)
    images.append(toP(out.squeeze()))
    plot(images, normalised=1)
    plt.show()
