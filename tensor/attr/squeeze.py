from tensor.main import tensor

def squeeze(self,dim=None) -> tensor:
  if isinstance(dim, tensor):
    if dim.ndim != 1:
      raise ValueError("dim passed as pyne.tensor must be 1 dimensional")
    else:
      dim = dim.data
  
  if isinstance(dim, float):
    raise ValueError("dim must only be Long int")
    
  nshape = self.shape.data
  ou =[]
  if dim == None:
    for i in nshape:
      if i != 1:
        ou.append(i)
    return self.view(ou)
    
  if isinstance(dim, tuple) or isinstance(dim, list):
    if max(dim) > len(nshape):
      raise IndexError(f"{self} has no dim '{max(dim)}'")
    else:
      nnshape=nshape
      for i in dim:
        if nshape[i] == 1:
          nnshape.pop(1)
      return self.view(nnshape)
  
  if isinstance(dim, int):
    if dim > self.ndim:
      raise IndexError(f"{self} has no dim '{dim}'")
    else:
      if nshape[dim] != 1:
        return self
      else:
        nshape.pop(dim)
        return self.view(nshape)
  else:
    raise ValueError(f"Unsupported argument of type '{type(dim)}'")