import torch
from tinygrad.tensor import Tensor as MyTensor
from torch import tensor as TorchTensor
a = MyTensor.randn((2, 3))
b = MyTensor.randn((3, 4))
c=a@b
# c=c.sum()
c.backward()

a_torch = TorchTensor(a.data, requires_grad=True)
b_torch = TorchTensor(b.data, requires_grad=True)
c_torch = a_torch@b_torch
# c_torch=c_torch.sum()
c_torch.backward(gradient=torch.ones_like(c_torch))

print("*"*100)
print(a.grad)
print(b.grad)
print("*"*100)

print(a_torch.grad)
print(b_torch.grad)
print("*"*100)



