from main import pyne
import torch
import time

a = pyne.randn((2,500,2))
print(a.mean(1, keepdim=False)) # ~0.
print(a. std(1, keepdim=False)) # ~1.