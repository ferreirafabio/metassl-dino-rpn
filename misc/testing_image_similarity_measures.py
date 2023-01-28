import argparse
import random

import matplotlib.pyplot as plt
from random import randint
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis as ERGAS
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from penalties import HSIM
from piqa.fsim import FSIM
from piqa.vsi import VSI
from piqa import gmsd, fsim, haarpsi, vsi
from kornia import enhance
import pprint
import pandas as pd

measures = [
    SSIM(),
    fsim.FSIM(),
    vsi.VSI(),
    haarpsi.HaarPSI(),
    HSIM(),
    nn.MSELoss(),
    ERGAS(),
    gmsd.GMSD(),
    PSNR(),
]
names = [
    "SSIM - max[0,1]",
    "FSIM - max[0,1]",
    "VSI - max[0,1]",
    "HaarPSI - max[0,1]",
    "HSIM - max[0,1]",
    "MSE - min[0,inf]",
    "ERGAS - min[0,inf]",
    "GMSD - min[0,inf]",
    "PSNR - max[0,inf]",
]
augmentations = {
    "RandAugment": transforms.Compose([
        transforms.RandAugment(),
        transforms.ToTensor(),
    ]),
    "RandomCrop0": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(32, scale=(0.05, 0.4)),
    ]),
    "RandomCrop1": transforms.Compose([
        transforms.RandomResizedCrop(32, (0.5, 1)),
        transforms.ToTensor()
    ]),
}
transf0 = transforms.Compose([
    transforms.RandomResizedCrop(32, (0.5, 1)),
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

toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()


def plot(imgs, args, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(num_rows, num_cols))
    fig.suptitle(f"Different Similarity Measures on {args.dataset}")
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0, 0].set(title='Original image')
    axs[0, 0].title.set_size(8)
    for col_idx in range(1, num_cols):
        axs[0, col_idx].set(title=str(col_idx))
        axs[0, col_idx].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def main(args):
    dataset = datasets.ImageFolder(args.data_path)
    images = []
    indices = []
    scores = [[] for _ in measures]
    if args.transform_type in augmentations.keys():
        transform = augmentations[args.transform_type]
    else:
        transform = random.choice(list(augmentations.values()))
        print(transform)
    for i in range(args.n_examples):
        idx = randint(0, len(dataset))
        indices.append(idx)
        image, _ = dataset.__getitem__(idx)
        loop = [image]
        for j in range(args.n_transforms):
            aug = transform(image).unsqueeze(0)
            target = toTensor(image).unsqueeze(0)
            for m, measure in enumerate(measures):
                loss = measure(aug, target).item()
                scores[m].append(loss)
            loop.append(toPIL(aug.squeeze()))
        images.append(loop)
    plot(images, args, indices)
    cols = [f"{im}-{idx}" for im in indices for idx in range(1, args.n_transforms + 1)]
    df = pd.DataFrame(scores, index=names, columns=cols)
    df.to_csv(args.output + ".csv")
    plt.savefig(args.output + ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluation of Image Similarity Measures')
    parser.add_argument("--output_dir", default=".", type=str)
    parser.add_argument("--output", default="out", type=str)
    parser.add_argument("--data_path", default="path/to/dataset/ImageFolder", type=str)
    parser.add_argument("--dataset", default="dataset", type=str)
    parser.add_argument("--n_examples", default=5, type=int)
    parser.add_argument("--n_transforms", default=5, type=int)
    parser.add_argument("--transform_type", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)
    argz = parser.parse_args()
    main(argz)
