import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from theta_norm import plot


if __name__ == "__main__":
    theta = torch.tensor([[[1, 0, 0.3], [0, 1, -0.5]]], dtype=torch.float)
    # theta = torch.rand(1, 2, 3)
    # theta[:, :, 2] = 0
    # theta[:, 0, 1] = theta[:, 0, 1].neg()
    theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)
    print(f"{theta=}")

    grid = F.affine_grid(theta, [1, 1, 32, 32], align_corners=True)
    print("corners: ", grid[0, 0, 0, :].tolist(), grid[0, 0, -1, :].tolist(), grid[0, -1, 0, :].tolist(), grid[0, -1, -1, :].tolist(), sep="\n", end="\n\n")

    a = torch.pow((grid[0, 0, 0, :] - grid[0, 0, -1, :]) / 2, 2).sum().sqrt().item()
    b = torch.pow((grid[0, 0, 0, :] - grid[0, -1, 0, :]) / 2, 2).sum().sqrt().item()
    print("length of side a: ", a)
    print("length of side b: ", b)
    print("area of new image: ", a * b, end="\n\n")
    print("area with det.: ", torch.det(theta[:, :, :2]).abs().item(), end="\n\n")

    print((theta[:, :, 2].abs() / 2))
    area_of_txy = (1 - (theta[:, :, 2].abs() / 2)).prod().pow(2).item()
    print(f"{area_of_txy=}")

    toP = transforms.ToPILImage()
    toT = transforms.ToTensor()

    data_path = "../../datasets/CIFAR10"
    cifar = datasets.CIFAR10(data_path)
    images = []

    image, _ = cifar.__getitem__(0)
    images.append(image)
    img = toT(image).unsqueeze(0)
    out = F.grid_sample(img, grid, align_corners=True)
    images.append(toP(out.squeeze()))
    plot(images)
    plt.show()
