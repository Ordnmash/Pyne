a = pyne.tensor([[2.,3.,1.,4.],[3.,2.,1.,4.]], requires_grad=True)
b = pyne.tensor([2.,2.,3.,3.],                 requires_grad=True)
c = a + b 
d = c * 4 # tensor * int [element wise operation]
e = pyne.tensor([[2,4], [4,2], [3,1], [4,2]])   # shape pyne.Size([4,2])
f = d @ e                                       # output shape = pyne.Size([2,2]) after matrix multiplying
g = f.sum(dim=None)                             # sum to a 1d tensor of element of 1
g.backward()                                    # computes dG/dF, dG/dE, dG/dD, dG/dC, dG/dB, dG/dA
