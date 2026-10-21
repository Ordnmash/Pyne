from createData import crData
from tensor.represent import __repr__ as repr
from tensor.attr.strides import strides

class tensor:
  """   Tensor object aka Array object  """
  """Global attributes for tensor object"""
  __repr__ = repr


  def __init__(self, data, dtype='float', _prev=(),_Strg=None,_Shape=None, _crd=True):
    """
    Construct tensor with their attributes
    * tensor accept mathematical datatypes including:
    - Bool, - 'inf' as float
    """
    if data is None: raise ValueError(f"tensor doesn't support elements of NoneType!")
    from main import pyne
    from size.Size import Size    
    
    self.dtype   = dtype
    self.data    = crData(data,self.dtype) if _crd else data
    self.children= set(_prev)

    self.shape    :Size  = Size(x=self.data, isshape=False) if _Shape is None else Size(_Shape,True)
    self.ndim     :int   = self.shape.ndim
    self.nelem    :int   = self.shape.nelem
    self.storage  :list  = pyne.storage(self.data) if _Strg is None else _Strg 
    self._strides :tuple = pyne.getStrides(self.shape.data)
    self.strides  :int   = strides
    self.offset   :int   = 0


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