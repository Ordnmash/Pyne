from main import pyne

x = pyne.tensor([
  [[0, 1, 2],
   [3, 4, 5]],

  [[6, 7, 8],
   [9, 10, 11]]
])

y = x.transpose(1, 2)

print("Original shape:", x.shape)
print("Original strides:", x._strides)

print("Transposed shape:", y.shape)
print("Transposed strides:", y._strides)

print("Storage shared:", y.storage is x.storage)

print("Transposed data:", y.data)