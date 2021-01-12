import os

import pandas as pd

from results.configs import *

tsf = pd.read_csv(DATALOADER_TXT, sep="\t", names=["long", "short", "temps"])
arr = np.array([])
for idx in range(len(tsf)):
    temps_img = os.path.join(ROOT_DIR, tsf.iloc[idx, 2])
    temps_img = pd.read_csv(temps_img, sep="\t", header=None)
    temps_img = temps_img.iloc[:, :-1]
    temps_img = temps_img.values
    temps_img = temps_img.flatten()
    arr = np.append(arr, temps_img)

print(f"Mean : {np.mean(arr)} \nStd: : {np.std(arr)}")

# Mean : 29.990661049107143
# Std: : 1.0492909714974938
