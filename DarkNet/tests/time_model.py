import timeit

setup = """
import torch
from models.vanilla_unet import VanillaUNet
from results.configs import SIZE
net = VanillaUNet().cuda()
low = torch.rand((1, 3, SIZE[0], SIZE[1])).cuda()
temps = torch.rand((1, 1, 32, 24)).cuda()
"""

stmt = """
net(low, temps)
"""

number = 100

print(
    f"Avg time for Vanilla net : {timeit.timeit(stmt=stmt, setup=setup, number=number) / number} secs"
)

# 1024 : 0.1809366957499992
# 512 : 0.04730567336000149
