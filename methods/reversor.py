def rev(x: list):
  out = []
  for i in range(len(x)):
    i -= len(x) # start from the end
    print(f"for x len of {len(x)}, i is {i}")
    out.append(x[i])
  return out