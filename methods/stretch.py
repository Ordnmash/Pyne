from tensor.main import tensor

def stretch(x: tensor,shape: list) -> tensor:
  if x.ndim != len(shape):
    x = x.todim(len(shape))
  if x.shape.data == shape:
    return x
   
  count = 0
  storage=x.storage
  
  for i in list(reversed(x.shape.data)):
    j = 0
    li_iterate = [l if l% i == 0 else None for l in range(1, len(storage)+1)]
    nliterate  = []
    for m in li_iterate:
      if m is not None:
        nliterate.append(m)
    istorage = []
    rshape   = list(reversed(shape))
    for k in nliterate:
      sjk = storage[j:k]
      if len(sjk) != rshape[count]:
        for _ in range(rshape[count]-1):
          sjk.append(sjk[0])
  
      istorage.append(sjk)
      j=k
    count += 1
    storage = istorage
  
  data = storage[0]
  return tensor(data ,x.dtype, (x,)) # storage && shape are unknown  