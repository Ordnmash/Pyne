from tensor.main import tensor

def view(self, *args) -> tensor:
      vout    = 1
      storage = self.storage
      shape   = []
      if isinstance(args[0], int): shape = list(args)
      if isinstance(args[0], list) or isinstance(args[0], tuple): shape = args[0]
      elif isinstance(args[0], tensor): shape = args[0].data
        
      for i in shape: 
        vout *= i
      if vout != self.nelem:
        raise ValueError(f"args of {shape} doesn't match to shape of {self.shape}")
      else:
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
        return tensor(storage, self.dtype, (self,))

