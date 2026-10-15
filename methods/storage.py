def storage(x) -> list:    
  if not isinstance(x, list):
    return [x]
  out = []
  def iterate(data: list):
    for i in data:
      if isinstance(i, list):
        iterate(i)
      else:
        for j in data:
          out.append(j)
        break
    return out
  
  out = iterate(x)
  return out