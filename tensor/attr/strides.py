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
    raise ValueError(f"item() can't be used for tensor with more than one elements.")
  return self.storage[0]