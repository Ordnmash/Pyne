def mean(self, dim=None, keepdim=False):
  dk  = self.shape.data[dim]
  out = self.sum(dim, keepdim) / dk
  return out