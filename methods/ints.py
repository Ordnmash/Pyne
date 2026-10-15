from tensor.main import tensor
import dtypes

def ones(shape: tuple, dtype=dtypes.Float):
  n = 1
  for s in shape:
    n*=s
  data = [1.0 for _ in range(n)] if dtype == 'float' else [1 for _ in range(n)]
  return tensor(data, dtype, _Strg=data.copy(), _Shape=[len(data)]).view(shape)

def zeros(shape: tuple, dtype=dtypes.Float):
  n = 1
  for s in shape:
    n*=s
  data = [1.0 for _ in range(n)] if dtype == 'float' else [1 for _ in range(n)]
  return tensor(data, dtype).view(shape)