from tensor.main import tensor
from tensor.operations.compatExtend import compatAndExtend as cae
from methods.stretch import stretch


def pow(self, other: tensor) -> tensor:
  
  other = other if isinstance(other, tensor) else tensor(other, self.dtype,_Strg=[other],_Shape=[])
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

  flat  = self .storage.copy()
  oflat = other.storage
  for i,d in enumerate(flat): flat[i] = d**oflat[i]
  
  return tensor(flat, self.dtype, (self,), _Strg=flat.copy(),_Shape=[len(flat)]).view(self.shape.data)