from tensor.main import tensor
from methods.stretch import stretch
from tensor.operations.compatExtend import compatAndExtend as cae


def mul(self, other: tensor) -> tensor:
  other = other if isinstance(other, tensor) else tensor(other, self.dtype)
  out = []
  dtype = 'long' if self.dtype == 'long' and other.dtype == 'long' else 'float'  
  nshape = cae(self, other)
  if not nshape:
    raise ValueError("shapes are incompatible for element-wise operation")
  
  if self.shape.data != nshape:
    self=stretch(self,nshape)
  if other.shape.data != nshape:
    other=stretch(other, nshape)

  self=self.view(1,self.nelem)
  other=other.view(1,other.nelem)
  
  out = [[]]
  for i in range(self.shape.data[1]):
    out[0].append(self.data[0][i] * other.data[0][i])
    
  return tensor(out,dtype=dtype, _prev=(self, other)).view(nshape)