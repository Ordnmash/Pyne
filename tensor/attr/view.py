# view
from tensor.main import tensor
from tensor.attr.nest import nest
from size.Size import Size

def view(self, *args) -> tensor:
  
  if len(args) == 1 and isinstance(args[0], tensor): shape = list(args[0].shape.data)
  elif len(args) == 1 and isinstance(args[0], (list, tuple)): shape = list(args[0])
  else: shape = list(args)

  if shape.count(-1) > 1: raise ValueError("only one dimension can be -1")
  if -1 in shape:
    known = 1
    for s in shape:
      if s != -1: known *= s
    if known == 0 or self.nelem % known: raise ValueError(f"cannot view {self.shape} as {shape}")
    shape[shape.index(-1)] = self.nelem // known

  n = 1
  for s in shape: n *= s
  if n != self.nelem: raise ValueError(f"cannot view {self.shape} as {shape}")

  out         = tensor.__new__(tensor)    # skip __init__: no crd, no re-flatten
  out.dtype   = self.dtype
  out.prev    = {self}
  out.storage = self.storage              # same list object, not a copy
  out.shape   = Size(shape, isshape=True)
  out.ndim    = len(shape)
  out.nelem   = self.nelem
  out.data    = nest(self.storage, shape) # built once, here
  return out