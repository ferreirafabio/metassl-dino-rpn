import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

warmup_epochs = 0
niter_per_ep = 100
start_warmup_value = 0
base_value = 10
epochs = 100
final_value = 100
cycles = 1

# warmup_schedule = np.array([])
# warmup_iters = warmup_epochs * niter_per_ep
# if warmup_epochs > 0:
#     warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

# iters = np.arange(epochs * niter_per_ep - warmup_iters)
iters = np.arange(epochs * niter_per_ep)
schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos((2 * cycles + 1) * np.pi * (iters / len(iters))))

# schedule = np.concatenate((warmup_schedule, schedule))

x = np.arange(len(schedule))
plt.plot(x, schedule)
plt.show()
