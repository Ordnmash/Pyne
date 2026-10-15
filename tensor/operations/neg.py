from tensor.main import tensor

def neg(self) -> tensor:
  flat = self.storage.copy()
  for i,d in enumerate(flat): flat[i] = -d
  return tensor(flat, self.dtype, (self,),_Strg=flat.copy(), _Shape=[len(flat)]).view(self.shape.data)