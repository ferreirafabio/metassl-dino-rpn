import time

import matplotlib.pyplot as plt

# from stn import STN
from stn import STN
from penalty_losses import ThetaCropsPenalty
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# import vision_transformer as vits

toP = transforms.ToPILImage()
toT = transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == "__main__":
    data_path = "../../datasets/CIFAR10"
    cifar = datasets.CIFAR10(data_path, transform=transform)
    loader = DataLoader(
        cifar,
        batch_size=1024,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )

    model = STN(invert_gradients=True, mode='translation_scale').cuda()
    criterion = ThetaCropsPenalty(invert=True).cuda()
    # criterion2 = nn.CrossEntropyLoss().cuda()
    # vit = vits.vit_tiny(16).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(10):
        print("Starting epoch ", epoch + 1)
        start = time.time()
        for batch, (images, targets) in enumerate(loader):
            images = images.cuda()
            # targets = targets.cuda()
            images, thetas = model(images)
            loss = 0
            for t in thetas[:2]:
                loss += criterion(theta=t, crops_scale=(0.4, 1))
            for t in thetas[2:]:
                loss += criterion(theta=t, crops_scale=(0.05, 0.4))
            losses.append(loss.item())

            # logits = vit(images[0])
            # loss2 = loss + criterion2(logits, targets)
            # optimizer.zero_grad()
            # loss2.backward()
            loss.backward()
            optimizer.step()

            # if batch % 100 == 0:
            #     end = time.time()
            #     print("time:", end-start)
            #     print("loss: ", loss.item())
            #     start = time.time()
        print(thetas[0][0].detach())
        print(thetas[2][0].detach())

    x = list(range(len(losses)))
    plt.plot(x, losses)
    plt.show()
