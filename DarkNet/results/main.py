import gc
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.loader import Precious
from data.transforms import simple_transform
from models.vanilla_unet import VanillaUNet


def main():
    net = VanillaUNet().cuda()
    trainset = Precious(simple_transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    criterion = nn.L1Loss()
    optim = Adam(net.parameters(), 1e-4)
    epochs = 150

    for epoch in range(epochs):
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            short = data['short_img']
            temps = data['temps_img'].unsqueeze(1)
            long = data['long_img']

            optim.zero_grad()

            out = net(short, temps)
            loss = criterion(out, long)
            gc.collect()
            loss.backward()
            optim.step()
            running_loss += loss
            running_loss += loss.item()
            if i % 18 == 17:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 18))
                running_loss = 0.0

    torch.save(net.state_dict(), f'runs/{time.time()}.pth')


if __name__ == '__main__':
    main()
