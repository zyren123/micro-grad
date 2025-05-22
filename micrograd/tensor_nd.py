import numpy as np
from typing import List, Tuple, Union, Set, Callable, Optional, Any

class TensorND:
    """
    支持多维张量操作的自动微分类，类似于PyTorch的Tensor
    """
    def __init__(self, 
                 data: Union[float, int, List, np.ndarray], 
                 shape: Optional[Tuple[int, ...]] = None, 
                 _children: Tuple["TensorND", ...] = (), 
                 _op: str = ''):
        # 转换数据为numpy数组
        if isinstance(data, (int, float)):
            data = np.array(data)
        elif isinstance(data, list):
            data = np.array(data)
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"数据类型必须是数字、列表或numpy数组，而不是{type(data)}")
            
        # 如果提供了shape，则将数据重塑为指定形状
        if shape is not None:
            try:
                data = data.reshape(shape)
            except ValueError:
                raise ValueError(f"无法将数据重塑为形状{shape}")
        
        self.data = data
        self.grad = np.zeros_like(data, dtype=float)
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
        self._prev = set(_children)
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """返回张量的形状"""
        return self.data.shape
    
    @property
    def size(self) -> int:
        """返回张量的元素总数"""
        return self.data.size
    
    @property
    def ndim(self) -> int:
        """返回张量的维度数"""
        return self.data.ndim
    
    def reshape(self, shape: Tuple[int, ...]) -> "TensorND":
        """重塑张量的形状"""
        out = TensorND(self.data.reshape(shape), _children=(self,), _op='reshape')
        
        def _backward():
            self.grad += out.grad.reshape(self.shape)
            
        out._backward = _backward
        return out
    
    def __add__(self, other) -> "TensorND":
        if not isinstance(other, TensorND):
            other = TensorND(other)
            
        # 尝试广播
        try:
            out_data = self.data + other.data
        except ValueError:
            raise ValueError(f"形状不兼容：{self.shape} 和 {other.shape}")
            
        out = TensorND(out_data, _children=(self, other), _op='+')
        
        def _backward():
            # 处理广播情况下的梯度累加
            if self.shape == out.shape:
                self.grad += out.grad
            else:
                # 为广播的维度求和梯度
                axes = tuple(i for i, (a, b) in enumerate(zip(out.shape[::-1], self.shape[::-1])) 
                            if a != b)
                self.grad += np.sum(out.grad, axis=axes).reshape(self.shape)
                
            if other.shape == out.shape:
                other.grad += out.grad
            else:
                # 为广播的维度求和梯度
                axes = tuple(i for i, (a, b) in enumerate(zip(out.shape[::-1], other.shape[::-1])) 
                            if a != b)
                other.grad += np.sum(out.grad, axis=axes).reshape(other.shape)
                
        out._backward = _backward
        return out
    
    def __mul__(self, other) -> "TensorND":
        if not isinstance(other, TensorND):
            other = TensorND(other)
            
        # 尝试广播
        try:
            out_data = self.data * other.data
        except ValueError:
            raise ValueError(f"形状不兼容：{self.shape} 和 {other.shape}")
            
        out = TensorND(out_data, _children=(self, other), _op='*')
        
        def _backward():
            # 处理广播情况下的梯度累加
            if self.shape == out.shape:
                self.grad += other.data * out.grad
            else:
                # 为广播的维度求和梯度
                grad_contrib = other.data * out.grad
                axes = tuple(i for i, (a, b) in enumerate(zip(out.shape[::-1], self.shape[::-1])) 
                            if a != b)
                self.grad += np.sum(grad_contrib, axis=axes).reshape(self.shape)
                
            if other.shape == out.shape:
                other.grad += self.data * out.grad
            else:
                # 为广播的维度求和梯度
                grad_contrib = self.data * out.grad
                axes = tuple(i for i, (a, b) in enumerate(zip(out.shape[::-1], other.shape[::-1])) 
                            if a != b)
                other.grad += np.sum(grad_contrib, axis=axes).reshape(other.shape)
                
        out._backward = _backward
        return out
    
    def matmul(self, other: "TensorND") -> "TensorND":
        """矩阵乘法操作"""
        if not isinstance(other, TensorND):
            other = TensorND(other)
            
        # 检查维度是否符合矩阵乘法要求
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("矩阵乘法需要至少一维的张量")
            
        try:
            out_data = np.matmul(self.data, other.data)
        except ValueError:
            raise ValueError(f"形状不兼容的矩阵乘法：{self.shape} 和 {other.shape}")
            
        out = TensorND(out_data, _children=(self, other), _op='matmul')
        
        def _backward():
            # 矩阵乘法的梯度计算
            if self.ndim == 1 and other.ndim == 1:
                # 向量点乘
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif self.ndim == 1:
                # 向量 @ 矩阵
                self.grad += np.matmul(out.grad, other.data.T)
                other.grad += np.outer(self.data, out.grad)
            elif other.ndim == 1:
                # 矩阵 @ 向量
                self.grad += np.outer(out.grad, other.data)
                other.grad += np.matmul(self.data.T, out.grad)
            else:
                # 矩阵 @ 矩阵
                self.grad += np.matmul(out.grad, other.data.T)
                other.grad += np.matmul(self.data.T, out.grad)
                
        out._backward = _backward
        return out
    
    def __pow__(self, other) -> "TensorND":
        """幂运算"""
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        out = TensorND(self.data ** other, _children=(self,), _op='**')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
            
        out._backward = _backward
        return out
    
    def relu(self) -> "TensorND":
        """ReLU激活函数"""
        out = TensorND(np.maximum(0, self.data), _children=(self,), _op='relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
            
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False) -> "TensorND":
        """沿指定轴求和"""
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = TensorND(out_data, _children=(self,), _op='sum')
        
        def _backward():
            # 创建与原始形状相同的梯度数组
            if axis is None:
                grad_expanded = np.ones_like(self.data) * out.grad
            else:
                grad_expanded = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                # 广播到原始形状
                grad_expanded = np.broadcast_to(grad_expanded, self.shape)
                
            self.grad += grad_expanded
            
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False) -> "TensorND":
        """沿指定轴求平均"""
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = TensorND(out_data, _children=(self,), _op='mean')
        
        def _backward():
            # 创建与原始形状相同的梯度数组
            if axis is None:
                grad_expanded = np.ones_like(self.data) * out.grad / self.size
            else:
                grad_expanded = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                # 计算轴上的元素数量
                if isinstance(axis, (list, tuple)):
                    axis_size = 1
                    for ax in axis:
                        axis_size *= self.shape[ax]
                else:
                    axis_size = self.shape[axis] if axis is not None else 1
                # 广播到原始形状并除以元素数量
                grad_expanded = np.broadcast_to(grad_expanded, self.shape) / axis_size
                
            self.grad += grad_expanded
            
        out._backward = _backward
        return out
    
    def backward(self):
        """反向传播计算梯度"""
        queue=[self]
        while queue:
            node=queue.pop(0)
            if node._backward is None:
                continue
            node._backward()
            queue.extend(node._prev)
        
            
    def __neg__(self):
        """负号运算符"""
        return self * -1
        
    def __radd__(self, other):
        """反向加法"""
        return self + other
        
    def __sub__(self, other):
        """减法"""
        return self + (-other)
        
    def __rsub__(self, other):
        """反向减法"""
        return TensorND(other) + (-self)
        
    def __rmul__(self, other):
        """反向乘法"""
        return self * other
        
    def __truediv__(self, other):
        """除法"""
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        return self * (other ** -1)
        
    def __rtruediv__(self, other):
        """反向除法"""
        return TensorND(other) * (self ** -1)
    
    def __matmul__(self, other):
        """@运算符用于矩阵乘法"""
        return self.matmul(other)
    
    def __rmatmul__(self, other):
        """反向@运算符"""
        return TensorND(other).matmul(self)
    
    def __repr__(self):
        """字符串表示"""
        return f"TensorND(data={self.data}, shape={self.shape}, grad={self.grad})"
    
    @classmethod
    def zeros(cls, *shape) -> "TensorND":
        """创建全零张量"""
        return cls(np.zeros(shape))
    
    @classmethod
    def ones(cls, *shape) -> "TensorND":
        """创建全一张量"""
        return cls(np.ones(shape))
    @classmethod
    def ones_like(cls, other) -> "TensorND":
        """创建与另一个张量形状相同的全一张量"""
        return cls(cls.ones(other.shape))
    
    @classmethod
    def zeros_like(cls, other) -> "TensorND":
        """创建与另一个张量形状相同的全零张量"""
        return cls(cls.zeros(other.shape))
    
    @classmethod
    def randn(cls, *shape) -> "TensorND":
        """创建标准正态分布随机张量"""
        return cls(np.random.randn(*shape))
    
    @classmethod
    def rand(cls, *shape) -> "TensorND":
        """创建[0,1)均匀分布随机张量"""
        return cls(np.random.rand(*shape)) 
    
    @classmethod
    def randint(cls, *shape) -> "TensorND":
        """创建[0,1)均匀分布随机张量"""
        return cls(np.random.randint(*shape))
    
    @classmethod
    def randn_like(cls, other) -> "TensorND":
        """创建与另一个张量形状相同的标准正态分布随机张量"""
        return cls(cls.randn(other.shape))
    
    @classmethod
    def rand_like(cls, other) -> "TensorND":
        """创建与另一个张量形状相同的均匀分布随机张量"""
        return cls(cls.rand(other.shape))
    
    @classmethod
    def randint_like(cls, other) -> "TensorND":
        """创建与另一个张量形状相同的均匀分布随机整数张量"""
        return cls(cls.randint(other.shape))
    
    
