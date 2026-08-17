a = pyne.tensor([[2,3,1,4],[3,2,1,4]])
b = pyne.tensor([2,2,3,3])
c = a + b 
d = c * pyne.tensor(4.0)
e = pyne.tensor([[2,4], [4,2], [3,1], [4,2]])   # shape pyne.Size([4,2])
f = d @ e                                       # output shape = pyne.Size([2,2]) after matrix multiplying
g = f.sum()                                     # sum to a 1d tensor of element of 1
g.backward()                                    # computes dG/dF, dG/dE, dG/dD, dG/dC, dG/dB, dG/dA
