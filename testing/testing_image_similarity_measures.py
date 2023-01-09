import sys

import matplotlib.pyplot as plt
from random import randint
import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from piqa.fsim import FSIM

transf0 = transforms.Compose([
    transforms.RandomResizedCrop(32, (0.5, 1), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
# second global crop
transf1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(32, scale=(0.05, 0.4)),
])

transf2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(180),
])
transf3 = transforms.Compose([
    transforms.RandAugment(),
    transforms.ToTensor()
])

topil = transforms.ToPILImage()
tens = transforms.ToTensor()
ssim = SSIM()
mse = nn.MSELoss()
psnr = PSNR()
# lpips = LPIPS()
ergas = ERGAS()
fsim = FSIM()


def hist_loss(pred, target):
    one = torch.histc(pred, 100, max=1)
    two = torch.histc(target, 100, max=1)
    base = torch.min(one, two)
    two[two == 0] = 1
    sim = torch.sum(torch.pow(base / two, 4)) / len(two)
    return sim


def hist_loss2(pred, target):
    one = torch.histc(pred, 100, max=1)
    two = torch.histc(target, 100, max=1)
    low = torch.min(one, two)
    high = torch.max(one, two)
    high[high == 0] = 1
    sim = torch.sum(torch.pow(low / high, 2)) / len(two)
    return sim


data_path = "../../datasets/CIFAR10"
cifar = datasets.CIFAR10(data_path)

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
    a.imshow(image)
    a.set_title("og")
    t_image = tens(image).unsqueeze(0)
    for i, deg in enumerate([0, 90, 180, 270], 1):
        a = ax[i]
        cropped = transf3(image).unsqueeze(0)
        sim1 = hist_loss(cropped, t_image).item() * 100
        sim2 = ssim(cropped, t_image).item() * 100
        tit = f'{sim1:.0f}||{sim2:.0f}'  # /{int(sim1)}/{int(sim2)}/{int(sim4)}'
        a.set_title(tit)
        a.axis('off')
        aaa = topil(cropped.squeeze())
        a.imshow(aaa)

plt.show()
