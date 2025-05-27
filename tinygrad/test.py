# from tensor import Tensor
from tensor import Tensor
from torch import tensor
if __name__ == "__main__":
    layer1=Tensor(3)
    layer2=Tensor(4)
    output=layer1+layer2
    output=output**2
    output.backward()
    print(output)
    print(layer1)
    print(layer2)
    
    
    print("*"*100+"Torch"+"*"*100)
    layer1=tensor(3.0,requires_grad=True)   
    layer2=tensor(4.0,requires_grad=True)
    output=layer1+layer2
    output=output**2
    output.backward()
    print(output.data)
    print(layer1.grad)
    print(layer2.grad)