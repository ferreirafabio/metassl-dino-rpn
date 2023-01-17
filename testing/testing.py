import torch
import torchvision
from PIL import Image
import PIL
import os

path = "../../datasets/CIFAR10"
dt = torchvision.datasets.CIFAR10(
    root=path,
    train=False,
    download=True,
)

path_cfr = path + "/val"
dt_cfr = torchvision.datasets.ImageFolder(
    root=path_cfr
)
print(len(dt_cfr))

for i in range(5):
    a, i = dt.__getitem__(i)
    for item in dt_cfr:
        b, j = item
        if i != j:
            continue
        if a == b:
            print("Found")
            break


# save_path = "../../datasets/CIFAR10/train/"
# for i in range(10):
#     os.mkdir(save_path + str(i))
# counter = [0] * 10
#
# for item in dt:
#     image, cls = item
#     n = counter[cls]
#     path = save_path + str(cls) + "/" + str(n) + ".png"
#     image.save(path,)
#     counter[cls] += 1
