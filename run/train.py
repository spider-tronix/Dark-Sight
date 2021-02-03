# OM NAMO NARAYANA
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import errno


batch_size = 2

sys.path.insert(1, "./")
from DarkNet.models.sid_unet import sidUnet
from DarkNet.models.lavi_unet import laviUnet
from data.darksight_dataloader import DarkSighDataLoader

#set to True if thermal response included
inc_therm = False

trainloader = DarkSighDataLoader(inc_therm=inc_therm).load(batch_size=batch_size)

model = laviUnet(inc_therm=inc_therm)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

plot_freq = 10
error_freq = 5

epochs = 100
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, gt = data
        optimizer.zero_grad()
        # print('gt.shape:', gt.shape)
        # forward + backward + optimize
        outputs = model(inputs)
        outputs = torch.transpose(outputs, 1, 3)
        outputs = torch.transpose(outputs, 1, 2)
        # print('output.shape', outputs.shape)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % batch_size == 1 and epoch % error_freq == 1:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(outputs.detach().numpy()[0])
            ax2.imshow(gt.detach().numpy()[0])
            ax3.imshow(torch.transpose(inputs.detach(), 1, 3).numpy()[0][:,:,3])

            os.chdir('./')

            save_dir = './results/lavi_unet_results/'

            if(not inc_therm):
                save_dir = './results/without_therm/'
                
            try:
                os.makedirs(save_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            plt.savefig(save_dir + "/epoch{}.png".format(epoch + 1))


print("Finished Training")
# plt.show()