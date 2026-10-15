from tensor.main import tensor

def sqrt(self):
  out = [i**0.5 for i in self.storage]
  return tensor(out, dtype='float', _prev=(self,)).view(self.shape.data)  