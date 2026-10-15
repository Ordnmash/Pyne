# tensor/nest.py
def nest(flat: list[any], shape: list[any]) -> list:
  """
  * From flat list of tensor elements
  * Build nested lists from a flat list
  * Returns MetaData for tensor
  """
  if not shape: return flat[0]
  steps = [1] * len(shape)               # elements per step at each dim
  for d in range(len(shape) - 2, -1, -1):
    steps[d] = steps[d + 1] * shape[d + 1]

  def build(off, d):
    if d == len(shape) - 1: return flat[off:off + shape[d]]
    return [build(off + i * steps[d], d + 1) for i in range(shape[d])]

  return build(0, 0)