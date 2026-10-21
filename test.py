from main import pyne
import torch

x = pyne.arange(0,64,1).view(2,4,4,2)

y = x @ x.view(4,2,2,4)
print(y.shape)