import timeit

setup = """
import torch
from models.vanilla_unet import VanillaUNet
"""

stmt = """
net = VanillaUNet().cuda()
input = torch.rand((1, 4, 512, 512)).cuda()
net(input)
"""

number = 1000

print(f'Avg time for Vanilla net : {timeit.timeit(stmt=stmt, setup=setup, number=number) / number} secs')
