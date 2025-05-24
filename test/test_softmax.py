import torch
from tinygrad.tensor import Tensor
import numpy as np
from torch import softmax, tensor as torch_Tensor


# 测试比较两种实现
if __name__ == "__main__":
    # 创建随机输入
    x_data = np.random.randn(3, 5)
    x = Tensor(x_data)
    
    # 使用我们的实现
    y1 = x.softmax()
    print("我们的softmax输出:")
    print(y1)
    
    # 创建与输出形状相同的全1梯度
    init_grad = np.ones_like(y1.data)
    print("初始梯度:")
    print(init_grad)
    
    # 使用相同的初始梯度
    # x.grad = None  # 确保梯度为空
    # y1.grad = init_grad
    y1.backward()
    grad1 = x.grad.copy() if x.grad is not None else None
    print("我们的softmax梯度:")
    print(grad1)
    
    print("*"*100)
    
    # 使用PyTorch实现
    x_torch = torch_Tensor(x_data, requires_grad=True)
    y2 = softmax(x_torch, dim=-1)
    print("PyTorch softmax输出:")
    print(y2)
    
    # 使用相同的初始梯度
    torch_init_grad = torch.tensor(init_grad)
    y2.backward(gradient=torch_init_grad)
    grad2 = x_torch.grad
    print("PyTorch softmax梯度:")
    print(grad2)
    
    # 比较结果
    print("\n输出差异:")
    print(np.max(np.abs(y1.data - y2.detach().numpy())))
    
    if grad1 is not None and grad2 is not None:
        print("\n梯度差异:")
        print(np.max(np.abs(grad1 - grad2.numpy())))
    else:
        print("梯度计算失败，无法比较")