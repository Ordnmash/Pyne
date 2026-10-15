def strides(self, dim:int) -> int:
  shape  = self.shape.data
  stride = 1
  if dim < 0:
    return stride
  for i in shape[dim+1:]:
    stride*=i
  return stride

def getStride(shape:list, dim:int) -> int:
    stride = 1
    if dim < 0:
      return stride
    for i in shape[dim+1:]:
      stride*=i
    return stride  

def item(self):
  if self.nelem > 1:
    raise ValueError(f"tensor with n-elements > 1, can't be converted to scalar!")
  return self.storage[0]