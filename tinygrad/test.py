# from tensor import Tensor
from tensor_reformat import Tensor

a=Tensor(3)
b=Tensor(4)
c=a+b
d=c+c
e=d+d
e.backward()
print(e.data)
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
print(e.grad)