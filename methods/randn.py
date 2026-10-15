from tensor.main import tensor
import random

def randn(shape: tuple):
  n = 1
  for s in shape: n*=s
  
  data = [random.uniform(-1, 1) for _ in range(n)]
  ts   = tensor(data, 'float', _Strg=data.copy()).view(shape)
  norm = (ts - ts.mean(-1, True)) / ts.std(-1, True)
  return norm

def randNorm(shape: tuple):
  a,b = -(shape[-1]**0.5), (shape[-1]**0.5)
  alld = 1
  for s in shape:
    alld*=s
  
  data = [random.uniform(a, b) for _ in range(alld)]
  ts   = tensor(data, 'float', _Strg=data.copy()).view(shape)
  norm = (ts - ts.mean(-1, True)) / ts.std(-1, True)
  return norm