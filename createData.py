from methods.storage import storage
from tensor.attr.nest import nest
from size.Size import Size

def crData(x: any, dtype='float'):
  if dtype == 'float':
    
    if isinstance(x, float) or isinstance(x, int): return float(x)
    elif isinstance(x, (list, tuple)): 
      return nest([float(s) for s in storage(x)], Size(x).data)

  if dtype == 'long':
      
      if isinstance(x, float) or isinstance(x, int): return int(x)
      elif isinstance(x, (list, tuple)): 
        return nest([int(s) for s in storage(x)], Size(x).data)

  if dtype == 'bool':
        
        if isinstance(x, float) or isinstance(x, int): return bool(x)
        elif isinstance(x, (list, tuple)): 
          return nest([bool(s) for s in storage(x)], Size(x).data)
