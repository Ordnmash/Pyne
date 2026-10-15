from tensor.main import tensor

def div(self, other):
  other = other if isinstance(other, tensor) else tensor(other, self.dtype)
  val   = (other**-1)
  return self * val