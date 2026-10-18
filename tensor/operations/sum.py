from tensor.main import tensor

def Sum(self, dim=None, keepdim=False):
  if dim is None:
    flat = [sum(self.storage)]
    if keepdim: 
          return tensor(flat, self.dtype, (self,), _Strg=flat.copy(), _Shape=[len(flat)]).todim(self.ndim)
    else: return tensor(flat, self.dtype, (self,), _Strg=flat.copy(), _Shape=[len(flat)])
  
  if dim >= self.ndim:
    raise IndexError(f"tensor of {self.ndim}dims has no dim '{dim}'")

  outshape      = self.shape.data.copy()
  outshape[dim] = 1

  flat  = []
  step  = self.strides(dim)
  outer = 1
  for k in outshape:outer *= k

  inner = self.shape.data[dim]
  base  = 0

  for o in range(outer):
    ihold = [self.storage[base + i * step] for i in range(inner)]
    flat.append(sum(ihold))
    base += 1

    if (o + 1) % step == 0: # jump in memory when outer iterate reaches steps
      base += (inner * step) - step

  outtens = tensor(flat,self.dtype, (self,), _Strg=flat.copy(), _Shape=[len(flat)]).view(outshape)
  return outtens if keepdim else outtens.squeeze(dim)