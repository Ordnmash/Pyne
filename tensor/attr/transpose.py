from tensor.main import tensor

def transpose(self, dim: int|tuple|None):
  # if dim is empty/None reverse dims.
  dimDict = {}
  def unpack(data, dim:int):
    for _ in range(dim):
      data = data[0]
    return data
  
  for d in dim:
    dimDict[d] = unpack(self.data, d)

  print(dimDict)
