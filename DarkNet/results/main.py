import gc
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.loader import Precious
from data.transforms import simple_transform
from models.vanilla_unet import VanillaUNet

lr_schedule = 0


def adjust_learning_rate(lr, optimizer, epoch):
    if lr_schedule == 0:
        lr = lr * (
            (0.2 ** int(epoch >= 60))
            * (0.2 ** int(epoch >= 120))
            * (0.2 ** int(epoch >= 160))
            * (0.2 ** int(epoch >= 220))
        )
    elif lr_schedule == 1:
        lr = lr * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif lr_schedule == 2:
        lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def main():
    net = VanillaUNet().cuda()
    trainset = Precious(simple_transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    lr = 1e-4
    criterion = nn.L1Loss()
    optim = Adam(net.parameters(), lr)
    epochs = 100

    for epoch in range(epochs):
        running_loss = 0
        # noinspection PyTypeChecker
        lr = adjust_learning_rate(lr, optim, epoch)

        for i, data in enumerate(trainloader, 0):
            short = data["short_img"]
            temps = data["temps_img"].unsqueeze(1)
            long = data["long_img"]

            optim.zero_grad()

            out = net(short, temps)
            loss = criterion(out, long)
            gc.collect()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            if i % 18 == 17:
                print(
                    "[%d, %5d] loss: %.3f lr: %f"
                    % (epoch + 1, i + 1, running_loss / 18, lr)
                )
                running_loss = 0.0
        if not (epoch + 1) % 20:
            torch.save(net.state_dict(), f"runs/{time.time()}.pth")

    torch.save(net.state_dict(), f"runs/{time.time()}.pth")


if __name__ == "__main__":
    main()
