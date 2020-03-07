import gc

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.loader import Precious
from data.transforms import simple_transform
from models.vanilla_unet import VanillaUNet


def main():
    net = VanillaUNet().cuda()
    trainset = Precious(simple_transform)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    dataset = trainset.__getitem__(140)

    # plt.imshow(np.hstack([dataset['short_img'].cpu().numpy().transpose((1, 2, 0)),
    #                       dataset['long_img'].cpu().numpy().transpose((1, 2, 0)),
    #                       ]))
    # plt.show()

    short = dataset['short_img'].unsqueeze(0)
    temps = dataset['temps_img'].unsqueeze(0).unsqueeze(0)
    long = dataset['long_img'].unsqueeze(0)

    criterion = nn.L1Loss().cuda()
    optim = Adam(net.parameters(), 1e-4)
    epochs = 1000

    for epoch in range(epochs):
        out = net(short, temps)
        loss = criterion(out, long)
        gc.collect()
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Epoch : {epoch} Loss : {loss.item()}")
        del loss
        del out


if __name__ == '__main__':
    main()
