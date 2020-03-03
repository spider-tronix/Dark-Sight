import numpy as np
import torch

SIZE = (512, 512)
DATALOADER_TXT = "/home/syzygianinfern0/sambashare/myFile_jpg.txt"
ROOT_DIR = "/home/syzygianinfern0/sambashare"

np.random.seed(0)
torch.manual_seed(0)
# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = False
