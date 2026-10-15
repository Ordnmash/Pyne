from tensor.main import tensor

def div(self, other):
  other = other if isinstance(other, tensor) else tensor(other, self.dtype,_Strg=[other],_Shape=[1])
  val   = (other**-1)
  return self * val