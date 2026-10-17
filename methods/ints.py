from  tensor.main import tensor

def ones(shape: list|tuple, dtype='long') -> tensor:
  
  n = 1
  for s in shape: n*=s
  flat = [1.0 for _ in range(n)] if dtype == 'float' else [1 for _ in range(n)]
  
  return tensor(flat, dtype, _Strg=flat.copy(), _Shape=[len(flat)], _crd=False).view(shape)

def zeros(shape: list|tuple, dtype='long') -> tensor:
  
  n = 1
  for s in shape:n*=s
  flat = [1.0 for _ in range(n)] if dtype == 'float' else [1 for _ in range(n)]
  
  return tensor(flat, dtype, _Strg=flat.copy(), _Shape=[len(flat)], _crd=False).view(shape)