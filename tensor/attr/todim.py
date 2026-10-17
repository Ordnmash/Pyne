from tensor.main import tensor

def todim(self: tensor,dim: int) -> tensor:
  if not isinstance(dim, int):
    raise AttributeError(f"tensor.todim() expects 'int' argument but got '{dim}'")
  
  change = dim - self.ndim
  if change <= 0: return self
  out        =    self.data
  
  for _ in range(change): out = [out]
  
  return tensor(out, dtype=self.dtype)