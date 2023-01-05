import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms


class HISTLoss(nn.Module):
    def __init__(self, bins=100, exponent=2, invert=False, **kwargs):
        super().__init__()
        self.bins = bins
        self.exponent = exponent
        self.invert = -1 if invert else 1

    def forward(self, output, target):
        target = torch.histc(torch.stack(target), bins=self.bins)
        loss = 0
        for itm in output:
            pred = torch.histc(itm, bins=self.bins)
            base = torch.min(pred, target)
            pred[pred == 0] = 1
            sim = torch.sum(torch.pow(base / pred, self.exponent)) / len(pred)
            step = 1 - sim
            loss += step
        return self.invert * loss


class SIMLoss(nn.Module):
    def __init__(self, resolution: int, min_sim: float = 1., invert=False, **kwargs):
        super().__init__()
        self.loss_fn = SSIM()
        self.resize = transforms.Resize(resolution)
        self.min_sim = 1 - min_sim
        self.invert = -1 if invert else 1

    def forward(self, output, target):
        target = self.resize(torch.stack(target))
        loss = 0
        for itm in output:
            step = 1 - self.loss_fn(self.resize(itm), target)
            step[step < self.min_sim] = 0
            loss += step
        return self.invert * loss
