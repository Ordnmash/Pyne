from  tensor.main import tensor

import random # python module for randomness

def randint(a:int,b:int, shape: tuple) -> tensor:
  nelem = 1
  flat  = []
  for s in shape:        nelem *= s
  for _ in range(nelem): flat.append(int(random.uniform(a,b)))
  
  return tensor(flat, 'long', _Strg=flat.copy(), _Shape=[len(flat)], _crd=False).view(shape)

def arange(start: int|float, end: int|float, step=1) -> tensor:
  if step == 0.0:
    raise ValueError(f"step cannot be zero!!!")
  if start > end and step > 0:
    raise ValueError(f"'start' must be less than 'end' when range dir is descent")
  elif start < end and step < 0:
    raise ValueError(f"'start' must be greater than 'end' when range dir is ascend")
  # 
  epochs = end - start
  if epochs % step != 0:
    raise ValueError(f"invalid step argument, {step} is not divisible for end - start.")
  flat = []
  if step > 0:
    ist = start-step
    for _ in range(int(epochs/step)):
      ist+=step
      flat.append(ist)
  else:
    ist = start-step
    for _ in range(int(epochs/step)):
      ist+=step
      flat.append(ist)

  return tensor(flat, 'float', _Strg=flat.copy(), _Shape=[len(flat)], _crd=False)