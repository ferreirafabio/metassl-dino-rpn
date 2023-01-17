import torch
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

ssim = SSIM(kernel_size=19)

path = "../../datasets/CIFAR10"
dt = datasets.CIFAR10(
    root=path,
    train=False,
    download=True,
)

t2t = transforms.ToTensor()
t2i = transforms.ToPILImage()
trcrop = transforms.RandomResizedCrop(size=32, scale=(0.9, 1))
t128 = transforms.Resize(128)
t256 = transforms.Resize(256)

figure = plt.figure()
figure.tight_layout()

pred = dt.__getitem__(11)[0]
plt.subplot(2, 1, 1)
plt.imshow(pred)
pred = t2t(pred).unsqueeze(dim=0)

target = trcrop(pred)  # 0.1 * pred
other = target.squeeze().permute(1, 2, 0)
# other = np.moveaxis(other, 0, -1)
plt.subplot(2, 1, 2)
plt.imshow(other)
# target = t2t(target).unsqueeze(dim=0)


print(ssim(pred, target))
plt.show()
