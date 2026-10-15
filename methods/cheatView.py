def cheatView(x:list, shape:list):
  storage = x
  for i in list(reversed(shape)):
    j = 0
    li_iterate = [l if l% i == 0 else None for l in range(1, len(storage)+1)]
    nliterate  = []
    for m in li_iterate:
      if m is not None:
        nliterate.append(m)
        
    istorage = []
    for k in nliterate:
      istorage.append(storage[j:k])
      j=k

    storage = istorage

  storage=storage[0]
  return storage
  