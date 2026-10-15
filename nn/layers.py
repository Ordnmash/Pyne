from tensor.main import tensor
import dtypes
from main import pyne

class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.bias = bias
    self.weight = pyne.randNorm([fan_in, fan_out])
    self.bias   = pyne.zeros([fan_in, fan_out], dtype=dtypes.Float) if self.bias else None

  def __call__(self, x):
    out = x @ self.weight
    if self.bias is not None:
      out += self.bias
    return out

  def parameters(self):
    return [self.weight] if self.bias is None else [self.weight, self.bias]