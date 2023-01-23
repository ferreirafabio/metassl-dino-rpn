import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from plot_script import plot


if __name__ == "__main__":
    toP = transforms.ToPILImage()
    toT = transforms.ToTensor()

    data_path = "../../datasets/CIFAR10"
    cifar = datasets.CIFAR10(data_path)

    image, _ = cifar.__getitem__(0)
    images = []
    img = toT(image).unsqueeze(0)

    ident = torch.tensor([[[1, 0, 0],
                           [0, 1, 0]]], dtype=torch.float)
    align = True
    angle = torch.tensor(torch.pi / 4)
    scale = 1
    rand = torch.rand(1, 2, 3)
    thetangle = torch.tensor([[[scale * torch.cos(angle), -scale * torch.sin(angle), 0],
                               [scale * torch.sin(angle), scale * torch.cos(angle), 0]]])
    lt = torch.tensor([-1, -1, 1], dtype=torch.float)
    rt = torch.tensor([1, -1, 1], dtype=torch.float)
    lb = torch.tensor([-1, 1, 1], dtype=torch.float)
    rb = torch.tensor([1, 1, 1], dtype=torch.float)

    """rotated by 45 degrees (counterclockwise)"""
    # === ALL FOLLOWING IMAGES ARE ALSO ROTATED BY 45 DEG === #
    theta = torch.tensor([[[torch.cos(angle), -torch.sin(angle), 0],
                           [torch.sin(angle), torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    images.append([toP(out.squeeze())])

    """DIAGONAL SCALED BY 1.5"""
    scale = 1.5
    theta = torch.tensor([[[scale * torch.cos(angle), -torch.sin(angle), 0],
                           [torch.sin(angle), scale * torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    # images[1].append(toP(out.squeeze()))

    """ALL PARAMETERS SCALED BY 1.5"""
    scale = 1.5
    theta = torch.tensor([[[scale * torch.cos(angle), -scale * torch.sin(angle), 0],
                           [scale * torch.sin(angle), scale * torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    # images[1].append(toP(out.squeeze()))print(theta)

    """DIAGONAL SCALED: X=1, Y=1.5"""
    scale = 1.
    scale2 = 1.5
    theta = torch.tensor([[[scale * torch.cos(angle), -torch.sin(angle), 0],
                           [torch.sin(angle), scale2 * torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    images[0].append(toP(out.squeeze()))

    """ROWS SCALED: 1ST=1.0, 2ND=1.5"""
    scale = 1.5
    theta = torch.tensor([[[torch.cos(angle), -torch.sin(angle), 0],
                           [scale * torch.sin(angle), scale * torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    images.append([toP(out.squeeze())])

    """COLS SCALED: 1ST=1.0, 2ND=1.5"""
    scale = 1.5
    theta = torch.tensor([[[torch.cos(angle), -scale * torch.sin(angle), 0],
                           [torch.sin(angle), scale * torch.cos(angle), 0]]])
    grid = F.affine_grid(theta, size=img.shape, align_corners=align)
    out = F.grid_sample(img, grid, align_corners=align)
    images[-1].append(toP(out.squeeze()))

    plot(images, normalised=True)
