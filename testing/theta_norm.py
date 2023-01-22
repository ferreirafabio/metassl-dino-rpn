import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import utils

if __name__ == "__main__":
    toTensor = transforms.ToTensor()
    toPIL = transforms.ToPILImage()

    data_path = "../../datasets/CIFAR10"
    cifar = datasets.CIFAR10(data_path)
    images = []

    theta = torch.tensor([[[1, 0, 0.25],
                           [0, 1, 0.25]]], dtype=torch.float)
    angle = torch.tensor(torch.pi/4)
    theta = torch.tensor([[[torch.cos(angle), -torch.sin(angle), 0.25],
                           [torch.sin(angle), torch.cos(angle), 0.25]]])

    x, _ = cifar.__getitem__(0)
    images.append(x)
    x = toTensor(x).unsqueeze(0)

    grid = F.affine_grid(theta, x.size(), True)
    out = F.grid_sample(x, grid, align_corners=True).squeeze()
    images.append(toPIL(out))

    normed = torch.tanh(theta)
    grid = F.affine_grid(normed, x.size())
    out = F.grid_sample(x, grid, align_corners=True).squeeze()
    images.append(toPIL(out))

    normed = theta / torch.linalg.norm(theta, ord=1, dim=2, keepdim=True)
    grid = F.affine_grid(normed, x.size(), align_corners=True)
    out = F.grid_sample(x, grid, align_corners=True).squeeze()
    images.append(toPIL(out))

    normed = theta / torch.linalg.norm(theta, ord=2, dim=2, keepdim=True)
    grid = F.affine_grid(normed, x.size())
    out = F.grid_sample(x, grid).squeeze()
    images.append(toPIL(out))

    labels = ['original', 'transformed', 'tanh', 'L1-norm', 'L2-norm']
    utils.plot(images, labels=labels)
    plt.show()

    """
    Other not working ideas
    """
    # # Create a mask to identify the black regions
    # mask = (out != 0).any(dim=0)
    # # Create the bounding box coordinates
    # coords = torch.nonzero(mask)
    # top_left = coords.min(dim=0)[0]
    # bottom_right = coords.max(dim=0)[0]
    # # Crop the image
    # cropped_result = out[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    # images.append(toPIL(cropped_result.squeeze()))
