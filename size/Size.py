class Size:
  def __init__(self, x:list, isshape=False):
    print(x, 'x for Size')
    if isinstance(x,type(None)): raise ValueError(f"cannot construct Size for NoneType!")
    
    self.x    = x
    self.data = self.getShape() if not isshape else list(x)
    self.ndim = len(self.data)
    self.nelem = self.getnelem()
    
  def __repr__(self):
    return f"pyne.Size({self.data})"
  
  def getShape(self):

    if isinstance(self.x, (int, float)):return []
     
    out = [len(self.x)]

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