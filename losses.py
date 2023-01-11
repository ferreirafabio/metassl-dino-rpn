import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms


def histogram_batch(
        input: torch.Tensor,
        bins: int = 30,
        min: float = 0.0,
        max: float = 1.0,
        kernel: str = "gaussian"
) -> torch.Tensor:
    """Estimates the histogram of the input.
    The calculation uses kernel density estimation. Default 'epanechnikov' kernel.

    Args:
        input: Input tensor to compute the histogram with shape :math:`(B, d1, d2, ...)`
        bins: The number of histogram bins.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive).
        kernel: kernel to perform kernel density estimation
          ``(`epanechnikov`, `gaussian`, `triangular`, `uniform`)``.
    Returns:
        Computed histogram of shape :math:`(B, bins)`.
    """
    if input is not None and not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not torch.Tensor. Got {type(input)}.")
    if not isinstance(bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(bins)}.")
    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')
    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=input.device, dtype=input.dtype) + 0.5)
    centers = centers.reshape(-1, 1, 1)
    u = torch.abs(input.flatten(1).unsqueeze(0) - centers) / delta

    if kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u**2)
    elif kernel in ("triangular", "uniform", "epanechnikov"):
        # compute the mask and cast to floating point
        mask = (u <= 1).to(u.dtype)
        if kernel == "triangular":
            kernel_values = (1.0 - u) * mask
        elif kernel == "uniform":
            kernel_values = torch.ones_like(u) * mask
        else:  # kernel == "epanechnikov"
            kernel_values = (1.0 - u**2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', 'uniform' or 'epanechnikov'. Got {kernel}.")
    hist = torch.sum(kernel_values, dim=-1).permute(1, 0)
    return hist


def histogram(
        input: torch.Tensor,
        bins: int = 30,
        min: float = 0.0,
        max: float = 1.0,
        kernel: str = "gaussian"
) -> torch.Tensor:
    """Estimates the histogram of the input.
    The calculation uses kernel density estimation. Default 'epanechnikov' kernel.

    Args:
        input: Input tensor to compute the histogram with shape :math:`(d1, d2, ...)`
        bins: The number of histogram bins.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive).
        kernel: kernel to perform kernel density estimation
          ``(`epanechnikov`, `gaussian`, `triangular`, `uniform`)``.
    Returns:
        Computed histogram of shape :math:`(B, bins)`.
    """
    if input is not None and not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not torch.Tensor. Got {type(input)}.")
    if not isinstance(bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(bins)}.")
    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')
    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=input.device, dtype=torch.half) + 0.5)
    centers = centers.reshape(-1, 1)
    u = torch.abs(input.flatten().unsqueeze(0) - centers) / delta
    # creates a (B x bins x (3 x H x W))-shape tensor

    if kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u**2)
    elif kernel in ("triangular", "uniform", "epanechnikov"):
        # compute the mask and cast to floating point
        mask = (u <= 1).to(u.dtype)
        if kernel == "triangular":
            kernel_values = (1.0 - u) * mask
        elif kernel == "uniform":
            kernel_values = torch.ones_like(u) * mask
        else:  # kernel == "epanechnikov"
            kernel_values = (1.0 - u**2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', 'uniform' or 'epanechnikov'. Got {kernel}.")

    hist = torch.sum(kernel_values, dim=-1)
    return hist


class HSIM(nn.Module):
    def __init__(self, exponent=1):
        super(HSIM, self).__init__()
        self.exponent = exponent

    def forward(self, pred, target):
        t = histogram_batch(target)
        p = histogram_batch(pred)
        m = torch.min(p, t)
        mask = (p == 0).to(p.dtype)
        p = p + mask
        score = torch.sum(torch.pow(m / p, self.exponent)) / t.shape.numel()
        return score


class HISTLoss(nn.Module):
    def __init__(self, bins=100, exponent=2, invert=False, **kwargs):
        super(HISTLoss).__init__()
        self.bins = bins
        self.exponent = exponent
        self.invert = -1 if invert else 1
        self.fn = HSIM()

    def forward(self, input, target):
        if isinstance(target, list):
            target = torch.stack(target)
        loss = 0
        for crop in input:
            score = self.fn(crop, target)
            step = 1 - score
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
