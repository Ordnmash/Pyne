from tensor.main import tensor

def todim(self,dim: int) -> tensor:
  if not isinstance(dim, int):
    raise AttributeError(f"tensor.todim() expects 'int' argument but got '{dim}'")
  change = dim - self.ndim
  out = self.data
  if change <= 0:
    return self
    
  for _ in range(change):
    out = [out]
  return tensor(out, dtype=self.dtype)