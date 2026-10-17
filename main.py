from tensor.main import tensor as ts

from methods.storage import storage
from methods import totensor
from methods.randn import randn, randNorm
from methods import ints
from tensor.attr.strides import getStride

class pyne:
  tensor  = ts
  storage = storage
  Long    = 'long'
  Float   = 'float'
  Bool    = 'bool'
  randint = totensor.randint
  arange  = totensor.arange
  randNorm= randNorm
  randn   = randn
  zeros   = ints.zeros
  ones    = ints.ones
  getStride = getStride