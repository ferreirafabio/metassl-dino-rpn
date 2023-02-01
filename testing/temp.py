import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


import torch
import torch.nn as nn

model = nn.Linear(1, 1)
nn.init.constant_(model.weight.data, 0.5)
nn.init.constant_(model.bias.data, 0.5)

# Define a loss function
def loss_fn(y_pred, y_true):
  return ((y_pred - y_true)**2).mean()

# Generate some dummy data
X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# Compute the predictions
y_pred = model(X)

# Compute the loss
loss = loss_fn(y_pred, y)

# Compute the gradients with respect to the parameters
# loss.backward()

# Divide the loss by a constant value before updating the parameters

c = 10.0
loss = loss / c
loss.backward()

# The gradients have been scaled by 1/c
print(model.weight.grad)
print(model.bias.grad)
