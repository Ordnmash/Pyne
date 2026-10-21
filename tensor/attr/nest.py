# tensor/attr/nest.py

def nest(flat:list, shape:list, strides=None, offset=0) -> list[list[int]]:

  if strides is None:
    strides = [1] * len(shape)

    for d in range(len(shape) - 2, -1, -1):
      strides[d] = strides[d + 1] * shape[d + 1]

  def build(off, d):

    if d == len(shape):
      return flat[off]

    return [
      build(off + i * strides[d], d + 1)
      for i in range(shape[d])
    ]

  return build(offset, 0)