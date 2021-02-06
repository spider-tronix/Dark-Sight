# OM NAMO NARAYANA


"""run on base dir"""


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

sys.path.insert(1, "./")
from DarkNet.models.sid_unet import sidUnet
from DarkNet.models.lavi_unet import laviUnet
from data.darksight_dataloader import DarkSighDataLoader


"""Parameters"""

batch_size = 2

# set to True if thermal response included
inc_therm = False
load_weights = True

# change the directory for required model
checkpoint_dir = "./DarkNet/checkpoints/lavi_unet/"
save_dir = "./results/lavi_unet_results/"
raw_format = False
model = laviUnet(inc_therm=inc_therm, raw_format=raw_format)
plot_freq = 10
error_freq = 5
checkpoint_freq = 25
dataset_dir = input("enter path for dataset")

epochs = 500
lr = 1e-4


"""training code"""
os.chdir("./")
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()


"""creating necessary directories"""

if not raw_format:
    save_dir = save_dir + "jpg_format/"
    checkpoint_dir = checkpoint_dir + "jpg_format/"

if not inc_therm:
    save_dir = save_dir + "without_therm/"
    checkpoint_dir = checkpoint_dir + "without_therm/"

try:
    os.makedirs(save_dir)
    print(save_dir + " created")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(checkpoint_dir)
    print(checkpoint_dir + " created")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if not load_weights or os.listdir(checkpoint_dir) == []:
    if load_weights:
        print("no models saved")
    epoch_loaded = 0
else:
    checkpoint = torch.load(checkpoint_dir + os.listdir(checkpoint_dir)[-1])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_loaded = checkpoint["epoch"]
    print("model loaded")

trainloader = DarkSighDataLoader(
    inc_therm=inc_therm, raw_format=raw_format, dataset_dir=dataset_dir
).load(batch_size=batch_size)


for epoch in range(epoch_loaded, epoch_loaded + epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, gt = data
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.transpose(outputs, 1, 3)
        outputs = torch.transpose(outputs, 1, 2)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        """saving results"""
        if i % batch_size == 1 and epoch % error_freq == 1:
            print(
                "epoch: %d, index: %d loss: %.3f" % (epoch + 1, i + 1, running_loss / 2)
            )
            running_loss = 0.0
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(outputs.detach().numpy()[0])
            ax2.imshow(gt.detach().numpy()[0])
            ax3.imshow(
                torch.transpose(torch.transpose(inputs.detach(), 1, 3), 1, 2).numpy()[
                    0
                ][:, :, :3]
            )

            # print(
            #     "input shape: ",
            #     torch.transpose(torch.transpose(inputs.detach(), 1, 3), 1, 2)
            #     .numpy()[0]
            #     .shape,
            #     "min: ",
            #     torch.transpose(torch.transpose(inputs.detach(), 1, 3), 1, 2).min(
            #         1, keepdim=True
            #     )[0],
            #     "max: ",
            #     torch.transpose(torch.transpose(inputs.detach(), 1, 3), 1, 2).max(
            #         1, keepdim=True
            #     )[0],
            # )

            # print(
            #     "prediction shape: ",
            #     outputs.detach().numpy()[0].shape,
            #     "min: ",
            #     outputs.min(1, keepdim=True)[0],
            #     "max: ",
            #     outputs.max(1, keepdim=True)[0],
            # )

            plt.savefig(save_dir + "epoch{}_index{}.png".format(epoch + 1, i + 1))

        """saving checkpoints"""
        if i % batch_size == 1 and epoch % checkpoint_freq == 1:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_dir + str(int(epoch / checkpoint_freq) + 1) + ".tar",
            )

            print(
                "checkpoint %d_%d.tar saved."
                % (int(epoch / checkpoint_freq) + 1, i + 1)
            )


print("Finished Training")
# plt.show()
