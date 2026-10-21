from main import pyne

x = pyne.arange(0,12,1).view(4,3)

y = x.transpose(0,1)

print(x.shape, y.shape,'\n')
print(x,'\n')
print(y)