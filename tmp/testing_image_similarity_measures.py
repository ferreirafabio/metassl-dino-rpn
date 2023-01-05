import sys

import matplotlib.pyplot as plt
from random import randint
import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from piqa.fsim import FSIM

transf0 = transforms.Compose([
    transforms.RandomResizedCrop(32, (0.5, 1), interpolation=InterpolationMode.NEAREST),
])
# second global crop
transf1 = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.05, 0.4), interpolation=InterpolationMode.NEAREST),
])

transf2 = lambda x, y: transforms.functional.rotate(x, y)
transf3 = transforms.RandAugment()

topil = transforms.ToPILImage()
ssim = SSIM()
mse = nn.MSELoss()
psnr = PSNR()
lpips = LPIPS()
ergas = ERGAS()
fsim = FSIM()


def hist_loss(pred, target):
    one = torch.histc(pred, 100, max=1)
    two = torch.histc(target, 100, max=1)
    base = torch.min(one, two)
    two[two == 0] = 1
    sim = torch.sum(torch.pow(base / two, 4)) / len(two)
    return sim


data_path = "../../datasets/CIFAR10"
cifar = datasets.CIFAR10(data_path, transform=transforms.ToTensor())

# a = cifar.__getitem__(randint(0, 50000-1))[0]
# b = transf1(a, )
# # b = cifar.__getitem__(randint(0, 50000-1))[0]
# val = hist_loss(a, b)
# print(val)


size = 8
f, axes = plt.subplots(5, 5, figsize=(size, size))
f.tight_layout()
for ax in axes:
    image = cifar.__getitem__(randint(0, 50000-1))[0]
    a = ax[0]
    a.axis('off')
    a.imshow(topil(image))
    a.set_title("og")
    image = image.unsqueeze(dim=0)
    for i, deg in enumerate([0, 90, 180, 270], 1):
        a = ax[i]
        cropped = transf2(image, deg)
        cropped = transf1(image)
        # sim0 = abs(hist_loss(cropped, image).item()) * 100
        # sim1 = abs(ssim(cropped, image).item()) * 100
        sim2 = lpips(cropped, image).item()
        # sim4 = (1 - abs(lpips(cropped, image).item())) * 100
        tit = f'{sim2:.3f}'  # /{int(sim1)}/{int(sim2)}/{int(sim4)}'
        a.set_title(tit)
        a.axis('off')
        aaa = topil(cropped.squeeze())
        a.imshow(aaa)

plt.show()
