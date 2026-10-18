class Size:
  def __init__(self, x:list, isshape=False):
    self.x    = x
    self.data = self.getShape() if not isshape else list(x)
    self.ndim = len(self.data)
    self.nelem = self.getnelem()
    
  def __repr__(self):
    return f"pyne.Size({self.data})"
  
  def getShape(self):

    if isinstance(self.x, (int, float)):return []
     
    try:out = [len(self.x)]
    except TypeError: print(self.x, 'is self.x'); raise TypeError(f"errrrrrr")
    def iterate(data):
      for i in data:
        if type(i) == list:
          out.append(len(i))
          iterate(i)
          break
        else:
          return
          
    iterate(self.x)
    return out


  def getnelem(self):
    n = 1
    for i in self.data:
      n*=i
    return n

  def __len__(self):
    
    shape = self.data
    nsh   = 1
    for s in shape:
      nsh *= s
    return nsh

  def __getitem__(self,i: int):
    return self.data[i]