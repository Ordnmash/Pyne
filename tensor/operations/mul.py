from tensor.main import tensor
from methods.stretch import stretch
from tensor.operations.compatExtend import compatAndExtend as cae


def add(self, other: tensor) -> tensor:
  
  other  = other if isinstance(other, tensor) else tensor(other, self.dtype, _Strg=other)
  out    = []
  dtype  = 'float' if not self.dtype == 'long' and not other.dtype == 'long' else 'long'
  nshape = cae(self, other)
  
  if not nshape:
    raise ValueError("shapes are incompatible for element-wise operation")
  
  if self .shape.data != nshape: self =stretch(self, nshape)
  if other.shape.data != nshape: other=stretch(other,nshape)

  self=self.view(self.nelem)
  other=other.view(other.nelem)
  
  out = []
  for i in range(self.nelem):
    out.append(self.storage[i] * other.storage[i])
    
  return tensor(out,dtype=dtype, _prev=(self, other), _Strg=out.copy()).view(nshape)