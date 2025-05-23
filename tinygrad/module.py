from tensor import Tensor
from torch.nn import Linear as TorchLinear
from torch import tensor as TorchTensor
import torch
class Module:
    def __init__(self):
        self.params = []
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0
            
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
class Linear(Module):
    def __init__(self, nin: int, nout: int):
        super().__init__()
        self.w=Tensor.randn((nin,nout))
        self.b=Tensor.zeros((nout,))
        self.params=[self.w,self.b]
        
    def forward(self, x: Tensor) -> Tensor:
        return x@self.w+self.b
    
    def __repr__(self):
        return f"Linear(nin={self.w.shape[0]}, nout={self.w.shape[1]})"

if __name__ == "__main__":
    # 测试我们自己的实现
    print("测试我们的实现：")
    x=Tensor.randn((10,5))
    linear=Linear(5,10)
    my_output=linear(x)
    my_output.backward()
    
    print("我们的权重梯度形状:", linear.w.data.shape)
    print("我们的偏置梯度形状:", linear.b.grad.shape)
    print("我们的权重梯度:")
    print(linear.w.grad)
    print("我们的偏置梯度:")
    print(linear.b.grad)
    
    print("\n" + "*"*80 + "\n")
    
    # 测试PyTorch的实现
    print("测试PyTorch的实现：")
    torch_linear=TorchLinear(5,10)
    torch_linear.weight.data=TorchTensor(linear.w.data.transpose())
    torch_linear.bias.data=TorchTensor(linear.b.data.transpose())
    torch_output=torch_linear(TorchTensor(x.data))
    torch_output.backward(gradient=torch.ones_like(torch_output))
    
    print("PyTorch权重梯度形状:", torch_linear.weight.grad.T.shape)
    print("PyTorch偏置梯度形状:", torch_linear.bias.grad.shape)
    print("PyTorch权重梯度:")
    print(torch_linear.weight.grad.T)
    print("PyTorch偏置梯度:")
    print(torch_linear.bias.grad)
        