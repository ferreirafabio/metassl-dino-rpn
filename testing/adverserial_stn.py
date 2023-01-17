import matplotlib.pyplot as plt

from stn import STN
from penalty_losses import ThetaLoss
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
        batch_size=32,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=2,
    )

    model = STN().cuda()
    criterion = ThetaLoss(invert=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(10):
        print("Starting epoch ", epoch + 1)
        for batch, data in enumerate(loader):
            img, _ = data
            img = img.cuda()
            _, theta = model(img)
            loss = criterion(theta)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print("loss: ", loss.item())
        print(theta[0][0])

    x = list(range(len(losses)))
    plt.plot(x, losses)
    plt.show()
