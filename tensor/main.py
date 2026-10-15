from tensor.represent import __repr__ as repr
from tensor.createData import crData  as crd
from tensor.attr.strides import strides

class tensor:
  """   Tensor object aka Array object  """
  """Global attributes for tensor object"""
  __repr__ = repr
  strides  = strides

  def __init__(self, data, dtype='float', _prev=(),_Strg=None,_Shape=None):
    self.dtype   = dtype
    self.data    = crd(data,self.dtype)
    self.children= set(_prev)
    from main import pyne
    from size.Size import Size
    print(f"shape passed? {bool(_Shape)}")
    self.shape   = Size(self.data) if not _Shape else Size(_Shape,True)
    self.ndim    = self.shape.ndim
    self.nelem   = self.shape.nelem
    self.storage = pyne.storage(self.data) if not _Strg else _Strg 
    



from tensor.attr.view import view
from tensor.attr.todim import todim
from tensor.attr.strides import item
from tensor.attr.squeeze import squeeze
from tensor.operations.mean import mean
from tensor.operations.sqrt import sqrt
from tensor.attr.transpose import transpose
from tensor.operations.std import std
from tensor.operations.std import var
from tensor.operations.add import add
from tensor.operations.sub import sub
from tensor.operations.neg import neg
from tensor.operations.mul import mul
from tensor.operations.pow import pow
from tensor.operations.div import div
from tensor.operations.sum import Sum

tensor.transpose = transpose
tensor.squeeze = squeeze
tensor.todim   = todim
tensor.item    = item
tensor.view    = view
tensor.__add__ = add
tensor.__sub__ = sub
tensor.__neg__ = neg
tensor.__mul__ = mul
tensor.__pow__ = pow
tensor.__truediv__ = div
tensor.sum         = Sum
tensor.mean        = mean
tensor.sqrt        = sqrt
tensor.std         = std
tensor.var         = var