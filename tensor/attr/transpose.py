from tensor.main import tensor
from tensor.attr.nest import nest
from size.Size import Size


def transpose(self, dim0, dim1) -> tensor:

  ndim = self.ndim

  # Validate dimensions
  if not (-ndim <= dim0 < ndim):
    raise IndexError(f"Dimension {dim0} out of range")

  if not (-ndim <= dim1 < ndim):
    raise IndexError(f"Dimension {dim1} out of range")

  # Normalize negative dimensions
  dim0 %= ndim
  dim1 %= ndim

  # Copy metadata
  shape = list(self.shape.data)
  strides = list(self._strides)

  # Exchange dimensions
  shape[dim0], shape[dim1] = shape[dim1], shape[dim0]

  strides[dim0], strides[dim1] = (
    strides[dim1], strides[dim0]
  )


  out = tensor.__new__(tensor)

  out.dtype = self.dtype
  out.children = {self}

  # Share underlying storage
  out.storage = self.storage

  # Assign new metadata
  out.shape = Size(shape, isshape=True)
  out.ndim = len(shape)
  out.nelem = self.nelem

  out._strides = tuple(strides)
  out.offset = self.offset

  # Construct logical nested representation
  out.data = nest(
    out.storage,
    shape,
    strides,
    out.offset
  )

  return out