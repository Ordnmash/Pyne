from tensor.main import tensor
from tensor.attr.nest import nest


def transpose(self, dim0, dim1) -> tensor:

  ndim = self.ndim

  if not (-ndim <= dim0 < ndim):
    raise IndexError(f"Dimension {dim0} out of range")

  if not (-ndim <= dim1 < ndim):
    raise IndexError(f"Dimension {dim1} out of range")

  dim0 %= ndim
  dim1 %= ndim

  shape = list(self.shape.data)
  strides = list(self._strides)

  shape[dim0], shape[dim1] = shape[dim1], shape[dim0]

  strides[dim0], strides[dim1] = (
    strides[dim1], strides[dim0]
  )

  out = tensor(
    nest(self.storage, shape, strides, self.offset),
    self.dtype,
    (self,),
    _Strg=self.storage,
    _Shape=shape,
    _crd=False
  )

  # Preserve the actual memory layout
  out._strides = tuple(strides)
  out.offset = self.offset

  return out