# OM NAMO NARAYANA
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


batch_size = 2

sys.path.insert(1, "./")
from DarkNet.models.sid_unet import sidUnet
from augmentation_test import DarkSightDataset, DarkSighDataLoader

trainloader = DarkSighDataLoader().load(batch_size=batch_size)
print(type(trainloader))

model = sidUnet()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

plot_freq = 10
error_freq = 20

epochs = 100
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, gt = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # print('output shape:', outputs.shape, 'ground truth shape: ', gt.shape)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (
            i % batch_size == 1 and epoch % error_freq == 1
        ):  # print every 2 mini-batches
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0
            plt.figure()
            plt.imshow(outputs.detach().numpy()[0])

print("Finished Training")
plt.show()
