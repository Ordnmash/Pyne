from tensor.main import tensor as ts
from methods.storage import storage
from dtypes import Float, Long, Bool
from methods import totensor
from methods.randn import randn, randNorm
from methods import ints
from tensor.attr.strides import getStride

class pyne:
  tensor  = ts
  storage = storage
  Long    = Long
  Float   = Float
  Bool    = Bool
  randint = totensor.randint
  arange  = totensor.arange
  randNorm= randNorm
  randn   = randn
  zeros   = ints.zeros
  ones    = ints.ones
  getStride = getStride