import numpy as np
import matplotlib.pyplot as plt
from micrograd.tensor_nd import TensorND
import torch  # 导入PyTorch进行对比

def simple_tensor_operations():
    """简单的张量操作示例，与PyTorch对比"""
    print("\n==== 标量操作对比 ====")
    # 我们的实现
    a = TensorND(2.0)
    b = TensorND(3.0)
    c = a * b
    c.backward()
    
    # PyTorch实现
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch
    c_torch.backward()
    
    print("TensorND结果:")
    print(f"a = {a.data}, grad = {a.grad}")
    print(f"b = {b.data}, grad = {b.grad}")
    print(f"c = a * b = {c.data}")
    
    print("\nPyTorch结果:")
    print(f"a = {a_torch.data.item()}, grad = {a_torch.grad.item()}")
    print(f"b = {b_torch.data.item()}, grad = {b_torch.grad.item()}")
    print(f"c = a * b = {c_torch.data.item()}")
    
    print("\n==== 向量加法对比 ====")
    # 我们的实现
    x = TensorND([1.0, 2.0, 3.0])
    y = TensorND([4.0, 5.0, 6.0])
    z = x + y
    z.backward()
    
    # PyTorch实现
    x_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    z_torch = x_torch + y_torch
    z_torch.backward(torch.ones_like(z_torch))
    
    print("TensorND结果:")
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x + y = {z.data}")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    
    print("\nPyTorch结果:")
    print(f"x = {x_torch.data}")
    print(f"y = {y_torch.data}")
    print(f"z = x + y = {z_torch.data}")
    print(f"x.grad = {x_torch.grad}")
    print(f"y.grad = {y_torch.grad}")
    
    print("\n==== 向量点乘对比 ====")
    # 我们的实现
    x.grad = np.zeros_like(x.data)
    y.grad = np.zeros_like(y.data)
    dot_product = x.matmul(y)
    dot_product.backward()
    
    # PyTorch实现
    x_torch.grad.zero_()
    y_torch.grad.zero_()
    dot_product_torch = torch.dot(x_torch, y_torch)
    dot_product_torch.backward()
    
    print("TensorND结果:")
    print(f"x @ y = {dot_product.data}")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    
    print("\nPyTorch结果:")
    print(f"x @ y = {dot_product_torch.data.item()}")
    print(f"x.grad = {x_torch.grad}")
    print(f"y.grad = {y_torch.grad}")
    
    print("\n==== 矩阵乘法对比 ====")
    # 我们的实现
    A = TensorND([[1.0, 2.0], [3.0, 4.0]])
    B = TensorND([[5.0, 6.0], [7.0, 8.0]])
    A.grad = np.zeros_like(A.data)
    B.grad = np.zeros_like(B.data)
    C = A @ B
    C.backward()
    
    # PyTorch实现
    A_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    C_torch = A_torch @ B_torch
    C_torch.backward(torch.ones_like(C_torch))
    
    print("TensorND结果:")
    print(f"A @ B = \n{C.data}")
    print(f"A.grad = \n{A.grad}")
    print(f"B.grad = \n{B.grad}")
    
    print("\nPyTorch结果:")
    print(f"A @ B = \n{C_torch.data}")
    print(f"A.grad = \n{A_torch.grad}")
    print(f"B.grad = \n{B_torch.grad}")
    
    print("\n==== 矩阵求和对比 ====")
    # 我们的实现
    A.grad = np.zeros_like(A.data)
    s = A.sum()
    s.backward()
    
    # PyTorch实现
    A_torch.grad.zero_()
    s_torch = A_torch.sum()
    s_torch.backward()
    
    print("TensorND结果:")
    print(f"sum(A) = {s.data}")
    print(f"A.grad = \n{A.grad}")
    
    print("\nPyTorch结果:")
    print(f"sum(A) = {s_torch.data.item()}")
    print(f"A.grad = \n{A_torch.grad}")
    
    print("\n==== 沿轴求和对比 ====")
    # 我们的实现
    A.grad = np.zeros_like(A.data)
    s_axis = A.sum(axis=0)
    s_axis.backward()
    
    # PyTorch实现
    A_torch.grad.zero_()
    s_axis_torch = A_torch.sum(dim=0)
    s_axis_torch.backward(torch.ones_like(s_axis_torch))
    
    print("TensorND结果:")
    print(f"sum(A, axis=0) = {s_axis.data}")
    print(f"A.grad = \n{A.grad}")
    
    print("\nPyTorch结果:")
    print(f"sum(A, dim=0) = {s_axis_torch.data}")
    print(f"A.grad = \n{A_torch.grad}")
    
    print("\n==== ReLU激活函数对比 ====")
    # 我们的实现
    x_relu = TensorND([-2.0, -1.0, 0.0, 1.0, 2.0])
    x_relu.grad = np.zeros_like(x_relu.data)
    y_relu = x_relu.relu()
    y_relu.backward()
    
    # PyTorch实现
    x_relu_torch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y_relu_torch = torch.relu(x_relu_torch)
    y_relu_torch.backward(torch.ones_like(y_relu_torch))
    
    print("TensorND结果:")
    print(f"relu(x) = {y_relu.data}")
    print(f"x.grad = {x_relu.grad}")
    
    print("\nPyTorch结果:")
    print(f"relu(x) = {y_relu_torch.data}")
    print(f"x.grad = {x_relu_torch.grad}")
    
    print("\n==== 幂运算对比 ====")
    # 我们的实现
    x_pow = TensorND([1.0, 2.0, 3.0])
    x_pow.grad = np.zeros_like(x_pow.data)
    y_pow = x_pow ** 2
    y_pow.backward()
    
    # PyTorch实现
    x_pow_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_pow_torch = x_pow_torch ** 2
    y_pow_torch.backward(torch.ones_like(y_pow_torch))
    
    print("TensorND结果:")
    print(f"x ** 2 = {y_pow.data}")
    print(f"x.grad = {x_pow.grad}")
    
    print("\nPyTorch结果:")
    print(f"x ** 2 = {y_pow_torch.data}")
    print(f"x.grad = {x_pow_torch.grad}")

if __name__ == "__main__":
    print("===== 与PyTorch对比的张量操作示例 =====")
    simple_tensor_operations() 