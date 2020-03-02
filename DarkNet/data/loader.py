import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset


def read_loader_txt(path="/home/syzygianinfern0/sambashare/myFile_jpg.txt"):
    with open(path) as handler:
        tsf = list(zip(*(line.strip().split('\t') for line in handler)))
    return tsf


class Precious(Dataset):
    """The DarkSights blood and sweat"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    tsf = read_loader_txt()

    short, long, temps = tsf[0], tsf[1], tsf[2]
    print(short)
