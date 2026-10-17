from tensor.main import tensor
from methods.stretch import stretch
from tensor.operations.compatExtend import compatAndExtend as cae


def add(self, other: tensor) -> tensor:
  
  other  = other if isinstance(other, tensor) else tensor(other, self.dtype, _Strg=other)
  dtype  = 'float' if not self.dtype == 'long' and not other.dtype == 'long' else 'long'
  nshape = cae(self, other)
  
  if not nshape:
    raise ValueError("shapes are incompatible for element-wise operation")
  
  if self .shape.data != nshape: self =stretch(self, nshape)
  if other.shape.data != nshape: other=stretch(other,nshape)

  self=self.view(self.nelem)
  other=other.view(other.nelem)
  
  flat = []
  for i in range(self.nelem):
    flat.append(self.storage[i] + other.storage[i])
    
  return tensor(flat,dtype=dtype, _prev=(self, other), _Strg=flat.copy(),_Shape=[len(flat)]).view(nshape)