from tensor.main import tensor

def div(self, other):
  
  other = other if isinstance(other, tensor) else tensor(other, self.dtype,_Strg=[other],_Shape=[])
  val   = (other**tensor(-1.0, 'float',_Strg=[-1.0], _Shape=[],_crd=False))
  return self * val