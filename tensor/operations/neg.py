from tensor.main import tensor

def neg(self) -> tensor:
  data = self.storage
  for i,d in enumerate(data):
    data[i] = -d
  return tensor(data, self.dtype, (self,)).view(self.shape.data)