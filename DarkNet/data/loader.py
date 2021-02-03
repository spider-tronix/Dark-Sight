import os

import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

from data.transforms import simple_transform
from results.configs import *


class Precious(Dataset):
    """The DarkSights blood and sweat"""

    def __init__(self, transform=None):
        """
        Init all class variables

        :param transform: To be decided
        """
        self.tsf = self.read_loader_txt(DATALOADER_TXT)
        self.transform = transform
        self.root_dir = ROOT_DIR

    @staticmethod
    def read_loader_txt(path=DATALOADER_TXT):
        """
        Reads from the dataset_gen text file and creates the dataframe

        :param path: Path for the dataset contents text file
        :return: Dataframe with cols of 'long', 'short', 'temps'
        """
        tsf = pd.read_csv(path, sep="\t", names=["long", "short", "temps"])
        return tsf

    def __len__(self):
        return len(self.tsf)

    def __getitem__(self, idx):
        """
        Returns an item from the dataset

        :param idx: Index
        :return: A dictionary of the images of that index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        long_img = os.path.join(self.root_dir, self.tsf.iloc[idx, 0])
        short_img = os.path.join(self.root_dir, self.tsf.iloc[idx, 1])
        temps_img = os.path.join(self.root_dir, self.tsf.iloc[idx, 2])

        long_img = io.imread(long_img)
        short_img = io.imread(short_img)
        temps_img = pd.read_csv(temps_img, sep="\t", header=None)
        temps_img = temps_img.iloc[:, :-1]
        temps_img = temps_img.values

        if self.transform:
            long_img = self.transform(long_img)
            short_img = self.transform(short_img)
            temps_img = torch.Tensor((temps_img - 29.99) / 1.049)

        data_sample = {
            "long_img": long_img.cuda(),
            "short_img": short_img.cuda(),
            "temps_img": temps_img.cuda(),
        }

        return data_sample


if __name__ == "__main__":
    precious = Precious(transform=simple_transform)
    fig = plt.figure()
    for i in range(len(precious)):
        sample = precious[i]
        print(
            i,
            sample["long_img"].shape,
            sample["short_img"].shape,
            sample["temps_img"].shape,
        )
        if i == 3:
            break
