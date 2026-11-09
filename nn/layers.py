from main import pyne

class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.isbias:bool          = bias
    self.weight:pyne.tensor   = pyne.randNorm([fan_in, fan_out])
    self.bias:pyne.tensor|None= pyne.zeros([fan_in, fan_out], dtype=dtypes.Float) if self.bias else None

  def forward(self, x):
    out = x @ self.weight
    if self.bias is not None:
      out += self.bias
    return out

  def parameters(self):
    return [self.weight] if self.bias is None else [self.weight, self.bias]
