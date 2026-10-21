from tensor.main import tensor
from tensor.attr.nest import nest
from methods.storage import storage

def matmul(self, other) -> tensor:

  if not isinstance(other, tensor):
    raise ValueError(f"pyne.matmul support tensors only!")

  other  = other.transpose(other.ndim-2, other.ndim-1) if other.ndim > 1 else other
  selfX  =  self.shape.data[-1]
  otherX = other.shape.data[-1]
  if selfX != otherX:
    raise ValueError(f"Shape {self.shape} and {other.shape} [{selfX}" 
                     f"!= {otherX}] mismatch for matmul")

  self_mtData  = nest( self.storage, [int( self.nelem/ selfX), selfX])
  other_mtData = nest(storage(other.data), [int(other.nelem/otherX),otherX])
  print(f"self.shape:{self.shape} - self_mtData is:\n{self_mtData}")
  print("")
  print(f"after .trans other.shape:{other.shape} - other_mtData is:\n{other_mtData}")
  outshape = self.shape.data[:-1] + [otherX]

  # compute calculations on 2d nested lists [INTEPRETED]
  out = []
  for smtd in self_mtData:
    for omtd in other_mtData:
      _out = []
      for i in range(len(smtd)):
        _out.append(smtd[i] * omtd[i])
      out.append(sum(_out))

  outensor = tensor(
    out,'float',(self, other),
    out.copy(),[len(out)],
    False
  ).view(outshape) # view as output shape
  return outensor