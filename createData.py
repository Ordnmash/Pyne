import dtypes
from methods.storage import storage
from methods.cheatView import cheatView as cv
from size.Size import Size

def crData(x: any, dtype='float'):
  print(f"dtype passed is {dtype}")
  if isinstance(dtype,dtypes.Float):
    dtype = dtype.type
  if isinstance(dtype, dtypes.Long):
    dtype = dtype.type
  if isinstance(dtype, dtypes.Bool):
    dtype = dtype.type


  if dtype == 'float':
    if isinstance(x, float) or isinstance(x, int):
      return float(x)
    
    elif isinstance(x, list):
      shape = Size(x).data
      inn   = [float(s) for s in storage(x)]
      x = cv(inn, shape)
      return x

  if dtype == 'long':
      if isinstance(x, float) or isinstance(x, int):
        return int(x)
      
      elif isinstance(x, list):
        shape = Size(x).data
        inn   = [int(s) for s in storage(x)]
        x = cv(inn, shape)
        return x

  if dtype == 'bool':
    if isinstance(x, float) or isinstance(x, int):
      return bool(x)
    
    elif isinstance(x, list):
      shape = Size(x).data
      inn   = [bool(s) for s in storage(x)]
      x = cv(inn, shape)
      return x
