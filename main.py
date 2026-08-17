import math

class pyne:

  class tensor:
    
    def __init__(self,data,orig=None, _prev=()):
      self.data = data
      self.shape = pyne.Size.getShape(self.data)
      self.storage = pyne.storage(self.data)
      self.nelem   = len(self.storage)
      self.ndim    = len(self.shape.data)
      self.original= orig # for copying of tensor to pass gradients to original tensors
      self._prev   = set(_prev)
      
    
    def view(self, *args):
      vout = 1
      storage = self.storage
      new=[]
      if type(args[0]) == list or type(args[0]) == tuple:
        args = args[0]
      elif type(args[0]) == pyne.tensor or type(args[0]) == pyne.Size:
        args = args[0].data
        
      for i in args:
        vout *= i
      if vout != self.nelem:
        raise ValueError(f"args of {args} doesn't match to shape of {self.shape}")
      else:
        for i in list(reversed(args)):
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
        return pyne.tensor(storage, self)
    
    def __add__(self, other):
      out = []
      def compatAndExtend(self, other):
        if self.ndim != other.ndim:
          #increase the dimensions
          if self.ndim > other.ndim:
            other = other.todim(self.ndim)
          else:
            self = self.todim(other.ndim)
        shape=[]
        for i in range(self.ndim):
          if self.shape.data[i] == other.shape.data[i]:
            shape.append(self.shape.data[i])
          else:
            if self.shape.data[i] == 1 or other.shape.data[i] == 1:
              g = self.shape.data[i] if self.shape.data[i] > other.shape.data[i] else other.shape.data[i]
              shape.append(g)
            else:
              return
        return shape
      
      nshape = compatAndExtend(self, other)
      if not nshape:
        raise ValueError("shapes are incompatible for element-wise operation")
      
      if self.shape.data != nshape:
        self=pyne.stretch(self,nshape)
      if other.shape.data != nshape:
        other=pyne.stretch(other, nshape)

      self=self.view(1,self.nelem)
      other=other.view(1,other.nelem)
      
      out = [[]]
      for i in range(self.shape.data[1]):
        out[0].append(self.data[0][i] + other.data[0][i])
        
      return pyne.tensor(out,_prev=(self, other)).view(nshape)

    def __mul__(self, other):
      out = []
      def compatAndExtend(self, other):
        if self.ndim != other.ndim:
          #increase the dimensions
          if self.ndim > other.ndim:
            other = other.todim(self.ndim)
          else:
            self = self.todim(other.ndim)
        shape=[]
        for i in range(self.ndim):
          if self.shape.data[i] == other.shape.data[i]:
            shape.append(self.shape.data[i])
          else:
            if self.shape.data[i] == 1 or other.shape.data[i] == 1:
              g = self.shape.data[i] if self.shape.data[i] > other.shape.data[i] else other.shape.data[i]
              shape.append(g)
            else:
              return
        return shape
      
      nshape = compatAndExtend(self, other)
      if not nshape:
        raise ValueError("shapes are incompatible for element-wise operation")
      
      if self.shape.data != nshape:
        self=pyne.stretch(self,nshape)
      if other.shape.data != nshape:
        other=pyne.stretch(other, nshape)

      self=self.view(1,self.nelem)
      other=other.view(1,other.nelem)
      
      out = [[]]
      for i in range(self.shape.data[1]):
        out[0].append(self.data[0][i] * other.data[0][i])
        
      return pyne.tensor(out,_prev=(self, other)).view(nshape)

    def __neg__(self):
      data = self.storage
      for i,d in enumerate(data):
        data[i] = -d
      return pyne.tensor(data).view(self.shape)

    def __sub__(self, other):
      return self + (-other)
    
    def todim(self,dim):
      change = dim - self.ndim
      out = self.data
      if change <= 0:
        return self
      for _ in range(change):
        out = [out]
      return pyne.tensor(out, orig=self)

    def squeeze(self,dim=None):
      if type(dim) == pyne.tensor:
        if dim.ndim != 1:
          raise ValueError("dim passed as pyne.tensor should be 1 dimensional")
        else:
          dim = dim.data
      
      if type(dim) == float:
        raise ValueError("dim must only be Long int")
        
      nshape = self.shape.data
      ou =[]
      if type(dim) == type(None):
        for i in nshape:
          if i != 1:
            ou.append(i)
        return self.view(ou)
      if type(dim) == tuple or type(dim) == list:
        if max(dim) > len(nshape):
          raise IndexError(f"{self} has no dim '{max(dim)}'")
        else:
          nnshape=nshape
          for i in dim:
            if nshape[i] == 1:
              nnshape.pop(1)
          return self.view(nnshape)
      if type(dim) == int:
        if dim > self.ndim:
          raise IndexError(f"{self} has no dim '{max(dim)}'")
        else:
          if nshape[dim] != 1:
            return self
          else:
            nshape.pop(dim)
            return self.view(nshape)

      raise ValueError(f"Unsupported argument of type '{type(dim)}'")

    def sum(self, dim=None, keepdim=False):
      if dim == None:
        xout = sum(self.storage)
        if keepdim:
          return pyne.tensor(xout).todim(self.ndim)
        else:
          return pyne.tensor([xout])
      
      if dim > self.ndim:
        raise ValueError(f"tensor of {self.shape} has no dim '{dim}'")
      else:
        pass

    def __getitem__(self, i):
      self.i = i
      return self.data[i]
        
    def __repr__(self):
      width = max((len(str(x)) for x in self.storage), default=1)
    
      def format_tensor(obj, level=0):
    
        # Scalar
        if not isinstance(obj, list):
          return str(obj).rjust(width)
    
        # Empty tensor
        if len(obj) == 0:
          return "[]"
    
        # 1D tensor
        if not isinstance(obj[0], list):
          values = [str(x).rjust(width) for x in obj]
          return "[" + ", ".join(values) + "]"
    
        # Higher-dimensional tensor
        lines = []
    
        for i, item in enumerate(obj):
    
          formatted = format_tensor(item, level + 1)
    
          if i > 0:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
              lines.append("\n" * (level + 2))
            else:
              lines.append("\n")
    
            lines.append(" " * (len("pyne.tensor(") + level))
    
          lines.append(formatted)
    
          if i != len(obj) - 1:
            lines.append(",")
    
        return "[" + "".join(lines) + "]"
    
      return f"pyne.tensor({format_tensor(self.data)})"

  def randint(a:int,b:int, shape: tuple):
    nelem = 1
    for k in shape:
      nelem *= k
    storage = []
    for n in range(nelem):
      storage.append(int(random.uniform(a,b)))
    return pyne.tensor(storage).view(shape)

  
  class Size:
    def __init__(self,data):
      self.data = data
      
    def __repr__(self):
      return f"pyne.Size({self.data})"
    
    def getShape(data):
      out = []
      if type(data) == int or type(data) == float:
        return pyne.Size([])
      out.append(len(data))
      def iterate(data):
        for i in data:
          if type(i) == list:
            out.append(len(i))
            iterate(i)
            break
          else:
            return
      iterate(data)
      return pyne.Size(out)

    def __len__(self):
      shape = self.data
      nsh   = 1
      for s in shape:
        nsh *= s
      return nsh

    def __getitem__(self,i):
      return self.data[i]

  def stretch(x: pyne.tensor,shape: list):
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
    
    storage=storage[0]
    return pyne.tensor(storage,x)  
    
    
  def storage(x):
    out = []
    if not type(x) == list:
      return [x]
    data = x
    def iterate(data):
      for i in data:
        if type(i) == list:
          iterate(i)
        else:
          for j in data:
            out.append(j)
          break
    
    iterate(data)
    return out
