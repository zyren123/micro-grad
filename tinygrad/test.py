# from tensor import Tensor
from tensor_reformat import Tensor
from torch import tensor
if __name__ == "__main__":
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
    
    
    print("*"*100+"Torch"+"*"*100)
    a=tensor(3.0,requires_grad=True)
    b=tensor(4.0,requires_grad=True)
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