from main import pyne

a = pyne.randint(0, 20, (20,))
b = a.view(2, 5, 2)

print("a.data:", a.data)
print("b.data:", b.data)

print("a.storage:", a.storage)
print("b.storage:", b.storage)

print("Same storage object:", a.storage is b.storage)