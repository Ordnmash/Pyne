from tensor.main import tensor
from tensor.operations.compatExtend import compatAndExtend as cae
from methods.stretch import stretch


def pow(self, other: tensor) -> tensor:
  other = other if isinstance(other, tensor) else tensor(other, 'float')
  dtype = 'long' if self.dtype == 'long' and other.dtype == 'long' else 'float'
  if not self.ndim and not other.ndim:
    out = self.data ** other.data
    return tensor(out,dtype, (self, other))
  
  nshape = cae(self, other)
  if not nshape:
    raise ValueError("shapes are incompatible for element-wise operation")

  if self.shape.data != nshape:
    self=stretch(self,nshape)
  if other.shape.data != nshape:
    other=stretch(other, nshape)

  data  = self .storage
  odata = other.storage
  for i,d in enumerate(data):
    data[i] = d**odata[i]
  return tensor(data, self.dtype, (self,)).view(self.shape.data)