def compatAndExtend(x, y):
  if x.ndim != y.ndim:
    #increase the dimensions
    if x.ndim > y.ndim:
      y = y.todim(x.ndim)
    else:
      x = x.todim(y.ndim)
  
  shape=[]
  for i in range(x.ndim):
    if x.shape.data[i] == y.shape.data[i]:
      shape.append(x.shape.data[i])
    else:
      if x.shape.data[i] == 1 or y.shape.data[i] == 1:
        g = x.shape.data[i] if x.shape.data[i] > y.shape.data[i] else y.shape.data[i]
        shape.append(g)
      else:
        return
  return shape