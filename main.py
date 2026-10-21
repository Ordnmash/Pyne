from methods     import ints
from methods     import totensor
from tensor.main import tensor
from methods.storage     import storage
from methods.randn       import randn, randNorm
from tensor.attr.strides import getStrides

class pyne:
  tensor             = tensor
  storage    :list   = storage
  Long       :str    = 'long'
  Float      :str    = 'float'
  Bool       :str    = 'bool'
  randint    :tensor = totensor.randint
  arange     :tensor = totensor.arange
  randNorm   :tensor = randNorm
  randn      :tensor = randn
  zeros      :tensor = ints.zeros
  ones       :tensor = ints.ones
  getStrides :tuple  = getStrides