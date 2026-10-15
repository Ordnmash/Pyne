from main import pyne
import torch
import time

print("...Pyne...\n")

t0 = time.perf_counter()
a1 = pyne.randint(1, 5, (20000,))
a  = a1.view((2,5000,2))

t1 = time.perf_counter()
print(f"{t1-t0:.2f}s for a1 and a constructions")
print(a.shape)
print("")
print(((a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)).std(1, keepdim=True))
t2 = time.perf_counter() 
print(f"{t2-t1:.2f}s for norm and std")

print("==========================")
print("...Pytorch...\n")
t0 = time.perf_counter()
a1 = torch.randint(1, 5, (20000,))
a  = a1.view((2,5000,2)).float()

t1 = time.perf_counter()
print(f"{t1-t0:.2f}s for a1 and a constructions")
print(a.shape)
print("")
print(((a - a.mean(1, keepdim=True)) / a.std(1, keepdim=True)).std(1, keepdim=True))
t2 = time.perf_counter() 
print(f"{t2-t1:.2f}s for norm and std")