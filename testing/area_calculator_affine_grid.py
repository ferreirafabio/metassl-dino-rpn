import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from theta_norm import plot


if __name__ == "__main__":
    theta = torch.tensor([[[.9, 0, 0.5], [0, -.9, -0.3]]], dtype=torch.float)
    # theta = torch.rand(1, 2, 3)
    # theta[:, :, 2] = 0
    # theta[:, 0, 1] = theta[:, 0, 1].neg()
    theta = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True).clamp(min=1)

    print(f"{theta=}")
    print(f"{theta[:, :, :2]=}")
    grid = F.affine_grid(theta, [1, 1, 32, 32], align_corners=True)
    print("corners: ", grid[0, 0, 0, :].tolist(), grid[0, 0, -1, :].tolist(), grid[0, -1, 0, :].tolist(), grid[0, -1, -1, :].tolist(), sep="\n")
    print()
    a = torch.pow((grid[0, 0, 0, :] - grid[0, 0, -1, :]) / 2, 2).sum().sqrt().item()
    print("length of side a: ", a)
    b = torch.pow((grid[0, 0, 0, :] - grid[0, -1, 0, :]) / 2, 2).sum().sqrt().item()
    print("length of side b: ", b)
    print("area of new image: ", a * b)
    print("area with det.: ", torch.det(theta[:, :, :2]).abs().item())

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
