from tensor.main import tensor

def sub(self, other):
  other = other if isinstance(other, tensor) else tensor(other, self.dtype,_Strg=[other],_Shape=[])
  return self + (-other)