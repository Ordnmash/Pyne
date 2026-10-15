from tensor.main import tensor

def cheatChunks(data: list, dim:int):
  # data is raw self.data
  for _ in range(dim):
    data = data[0]
  return data


def matmul(self, other):
  other = other if isinstance(other, tensor) else tensor(other, self.dtype)
  if self.dtype == 'long' or other.dtype == 'long':
    raise TypeError(f"matmul works on float tensors! try float() to convert dtype.")

  if self.shape.data[-1] != other.shape[0]:
    raise ValueError(f"shapes of {self.shape}@{other.shape} are incompatible for matmul.")

  out = []
  A   = cheatChunks(self.data, self.ndim-1)
  B   = cheatChunks(other.data, 0)
  t   = len(A)
  tin = len(B)
  for ia in range(t):
    for ib in range(tin):
      tA = tensor(A[ia], 'float')
      tB = tensor(B[ib], 'float')
      out.append((tA*tB).sum())