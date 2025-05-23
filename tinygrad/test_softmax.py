import torch
from tensor import Tensor
import numpy as np
from torch import softmax,tensor as torch_Tensor


# 测试比较两种实现
if __name__ == "__main__":
    # 创建随机输入
    x = Tensor(np.random.randn(3, 5))
    
    # 使用手动梯度版本
    y1 = x.softmax()
    print(y1)
    y1.backward()
    grad1 = x.grad.copy() if x.grad is not None else None
    x.grad = None  # 重置梯度
    print(grad1)
    
    print("*"*100)
    x=torch_Tensor(x.data,requires_grad=True)
    y2=softmax(x,dim=-1)
    print(y2)
    y2.backward(gradient=torch.ones_like(y2))
    grad2=x.grad
    print(grad2)
    
    # # 比较结果
    # print("输出差异:")
    # print(np.max(np.abs(y1.data - y2.data)))
    
    # if grad1 is not None and grad2 is not None:
    #     print("梯度差异:")
    #     print(np.max(np.abs(grad1 - grad2)))
    # else:
    #     print("梯度计算失败，无法比较")