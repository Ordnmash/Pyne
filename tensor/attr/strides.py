# tensor/attr/strides.py

def getStrides(shape: list) -> tuple:
  """Calculate contiguous strides from a shape."""

  strides = [1] * len(shape)

  for i in range(len(shape) - 2, -1, -1):
    strides[i] = strides[i + 1] * shape[i + 1]

  return tuple(strides)


def strides(self, dim: int = None):
  """Return the tensor's stored strides."""

  if dim is None:
    return self._strides

  if not isinstance(dim, int):
    raise TypeError("dimension must be an integer")

  if not (-self.ndim <= dim < self.ndim):
    raise IndexError(f"dimension {dim} out of range")

  return self._strides[dim]

def item(self):
  if self.nelem > 1:
    raise ValueError(f"tensor with n-elements > 1, can't be converted to scalar!")
  return self.storage[0]