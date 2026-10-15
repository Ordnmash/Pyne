from tensor.main import tensor
from main        import pyne

def view(self, *args) -> tensor:
  shape   = args 
  if isinstance(shape[0], (tensor, tuple, list)): shape = shape[0]

  vout    = 1
  for s in shape: vout*=s
  if vout != self.nelem: raise ValueError(f"cannot view tensor of {self.shape} with {shape}")

  out     = []
  storage = self.storage.copy()