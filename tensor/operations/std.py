def var(self, dim=None, keepdim=False):

  n   = self.shape.data[dim] if dim is not None else self.nelem
  out = (((self - self.mean(dim, True))**2).sum(dim, keepdim) / n)
  return out

def std(self, dim=None, keepdim=False):
  return self.var(dim, keepdim).sqrt()