import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from kornia import enhance


def histogram(
        image: torch.Tensor,
        bins: int = 100,
        min: float = 0.0,
        max: float = 1.0,
        kernel: str = "gaussian"
) -> torch.Tensor:
    """Estimates the histogram of the input.

    Args:
        image:
    """
    delta = (max - min) / bins

    centers = min + delta * (torch.arange(bins, device=image.device, dtype=image.dtype) + 0.5)
    centers = centers.reshape(-1, 1)

    u = torch.abs(image.unsqueeze(0).flatten(2) - centers) / delta

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
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', " f"'uniform' or 'epanechnikov'. Got {kernel}.")

    hist = torch.sum(kernel_values, dim=(-2, -1)).permute(1, 2, 0)
    return hist


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.lin1 = nn.Conv2d(3, 6, 3, padding=1)
        self.lin2 = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, x):
        logits = self.lin2(F.relu(self.lin1(x), inplace=True))
        return logits


def check_params(model1, model2):
    for p1, p2 in zip(model1.items(), model2.items()):
        if p1[1].data.ne(p2[1]).sum() > 0:
            return "False"
    return "True"


shape = 1, 3, 32, 32
X = torch.rand(shape)

model = SimpleNet()
other = SimpleNet()
aa = deepcopy(model.state_dict())
bb = deepcopy(other.state_dict())

loss_fn = nn.MSELoss()
ssim = SSIM()
optimizer = torch.optim.SGD([{"params": model.parameters()},
                             {"params": other.parameters()}],
                            lr=1e-1)

optimizer.zero_grad()
out = model(X)
loss1 = 0
if 1 < 2:
    # fn = enhance.image_histogram2d
    one = histogram(X)
    two = histogram(out)
    base = torch.min(one, two)
    base[base == 0] = 1
    sim = torch.sum(torch.pow(base / two, 2)) / len(two)
    loss1 = 1 - sim
    print(loss1)
out2 = other(out)

loss2 = 0  # loss_fn(X, out2)
loss = loss1 + loss2
# print(aa)

print("============================================")
print("============================================")
print(loss1)
print("============================================")
print(loss2)
print("============================================")
# loss[loss > .5] = 0
print(loss)
loss.backward()
optimizer.step()
print("============================================")
print("============================================")
print(check_params(aa, model.state_dict()))
print("============================================")
print(check_params(bb, other.state_dict()))
print("============================================")
print("============================================")
