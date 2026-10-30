from tensor.main import tensor
from tensor.attr.nest import nest
from methods.storage import storage
from itertools import product
from math import prod


def broadcast_shape(a: list, b: list) -> list:

  result = []

  for x, y in zip(
    reversed(a),
    reversed(b)
  ):
    if x != y and x != 1 and y != 1:
      raise ValueError(
        f"Cannot broadcast batch dimensions {a} and {b}"
      )

    result.append(max(x, y))

  if len(a) > len(b):
    result.extend(reversed(a[:len(a)-len(b)]))

  elif len(b) > len(a):
    result.extend(reversed(b[:len(b)-len(a)]))

  return list(reversed(result))


def batch_index(coords: list, shape: list|None) -> int:

  if not shape:
    return 0

  coords = coords[-len(shape):]

  index = 0

  for coord, dim in zip(coords, shape):
    index = index * dim + (0 if dim == 1 else coord)

  return index


def matmul(self, other) -> tensor:

  if not isinstance(other, tensor):
    raise ValueError("pyne.matmul supports tensors only!")

  a_shape = list(self.shape.data)
  b_shape = list(other.shape.data)

  if self.ndim == 0 or other.ndim == 0:
    raise ValueError("matmul does not support scalar tensors")

  # Handle vector inputs
  a_vector = self.ndim == 1
  b_vector = other.ndim == 1

  if a_vector:
    a_shape = [1] + a_shape

  if b_vector:
    b_shape = b_shape + [1]

  # Matrix dimensions
  M = a_shape[-2]
  K = a_shape[-1]

  K2 = b_shape[-2]
  N = b_shape[-1]

  if K != K2:
    raise ValueError(
      f"Shape {self.shape} and {other.shape} "
      f"mismatch for matmul: {K} != {K2}"
    )

  # Batch dimensions
  a_batch = a_shape[:-2]
  b_batch = b_shape[:-2]

  batch_shape = broadcast_shape(a_batch, b_batch)

  # Output shape before vector adjustments
  outshape = batch_shape + [M, N]

  # Transpose other so columns become rows
  if b_vector:
    other_data = list(other.storage)
  else:
    other_t = other.transpose(other.ndim-2, other.ndim-1)
    other_data = storage(other_t.data)

  # Convert flat storage into 2D matrices
  a_matrices = nest(
    self.storage,
    [prod(a_shape[:-2]) * M, K]
  )

  b_matrices = nest(
    other_data,
    [prod(b_shape[:-2]) * N, K]
  )

  # Group rows into individual matrices
  a_matrices = [
    a_matrices[i:i+M]
    for i in range(0, len(a_matrices), M)
  ]

  b_matrices = [
    b_matrices[i:i+N]
    for i in range(0, len(b_matrices), N)
  ]

  # Compute batch coordinates
  batch_coords = (
    product(*(range(dim) for dim in batch_shape))
    if batch_shape
    else [()]
  )

  out = []

  for coords in batch_coords:

    ai = batch_index(coords, a_batch)
    bi = batch_index(coords, b_batch)

    a_matrix = a_matrices[ai]
    b_matrix = b_matrices[bi]

    # Matrix multiplication
    for smtd in a_matrix:
      for omtd in b_matrix:

        _out = []

        for i in range(K):
          _out.append(smtd[i] * omtd[i])

        out.append(sum(_out))

  # PyTorch vector shape rules
  if a_vector:
    outshape.pop(-2)

  if b_vector:
    outshape.pop(-1)

  # Construct result
  outensor = tensor(
    out,
    'float',
    (self, other),
    out.copy(),
    [len(out)],
    False
  ).view(outshape)

  return outensor
