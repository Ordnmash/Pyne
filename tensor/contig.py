def is_contiguous(self) -> bool:
  expected = 1

  for size, stride in zip(
    reversed(self.shape.data),
    reversed(self._strides)
  ):
    if size == 1:
      continue

    if stride != expected:
      return False

    expected *= size

  return True