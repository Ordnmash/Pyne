from tensor.main import tensor

def sub(self, other):
  other = other if isinstance(other, tensor) else tensor(other, dtype=self.dtype)
  return self + (-other)