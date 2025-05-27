import numpy as np
class Tensor:
    def __init__(self,data,_children=()):
        if not isinstance(data, np.ndarray):
            data=np.array(data)
        self.data = data
        self.shape = data.shape
        self.grad_fn = None
        self.grad = np.zeros(self.shape)
        self.children=set(_children)
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def _handle_broadcast_grad(self, grad, shape):
        """处理广播情况下的梯度计算
        
        Args:
            grad: 原始梯度
            shape: 需要匹配的形状
            
        Returns:
            处理后的梯度，形状与shape匹配
        """
        # 确保grad是numpy数组
        if not isinstance(grad, np.ndarray):
            grad = np.array(grad)
            
        result_grad = np.copy(grad)
        # 如果形状不同，需要对梯度进行求和以匹配原始张量形状
        if shape != grad.shape:
            # 按照numpy的广播规则，对多余的维度求和
            axes = tuple(range(len(grad.shape) - len(shape)))
            if axes:
                result_grad = np.sum(result_grad, axis=axes)
                
            # 对于大小为1的维度进行求和（广播维度）
            for i, dim in enumerate(shape):
                if dim == 1:
                    result_grad = np.sum(result_grad, axis=i, keepdims=True)
                    
        return result_grad
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        out=Tensor(self.data + other.data,_children=(self,other))
        def _ADD_backward():
            self.grad += self._handle_broadcast_grad(out.grad, self.shape)
            other.grad += self._handle_broadcast_grad(out.grad, other.shape)
        out._backward = _ADD_backward
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        out=Tensor(self.data * other.data,_children=(self,other))
        def _MUL_backward():
            self.grad += self._handle_broadcast_grad(out.grad * other.data, self.shape)
            other.grad += self._handle_broadcast_grad(out.grad * self.data, other.shape)
        out._backward = _MUL_backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        out=Tensor(self.data ** other,_children=(self,))
        def _POW_backward():
            self.grad += self._handle_broadcast_grad(out.grad * other * (self.data ** (other - 1)), self.shape)
        out._backward = _POW_backward
        return out

    def __matmul__(self, other):
        out=Tensor(self.data @ other.data,_children=(self,other))
        def _MATMUL_backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _MATMUL_backward
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        return other * (self ** -1)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def relu(self):
        out=Tensor(np.maximum(0, self.data),_children=(self,))
        def _RELU_backward():
            self.grad += self._handle_broadcast_grad(out.grad * (self.data > 0), self.shape)
        out._backward = _RELU_backward
        return out
    
    def exp(self):
        out=Tensor(np.exp(self.data),_children=(self,))
        def _EXP_backward():
            self.grad += self._handle_broadcast_grad(out.grad * np.exp(self.data), self.shape)
        out._backward = _EXP_backward
        return out
    
    def sum(self):
        out=Tensor(np.sum(self.data),_children=(self,))
        def _SUM_backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _SUM_backward
        return out
    
    def mean(self):
        out=Tensor(np.mean(self.data),_children=(self,))
        def _MEAN_backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size
        out._backward = _MEAN_backward
        return out

    def log(self):
        out=Tensor(np.log(self.data),_children=(self,))
        def _LOG_backward():
            self.grad += self._handle_broadcast_grad(out.grad / self.data, self.shape)
        out._backward = _LOG_backward
        return out

    def softmax(self, dim=-1):
        """计算softmax
        
        Args:
            dim: 计算softmax的维度，默认为最后一个维度
            
        Returns:
            softmax结果的Tensor
        """
        # 为数值稳定性，减去最大值
        x_max = np.max(self.data, axis=dim, keepdims=True)
        shifted = self.data - x_max
        exp_vals = np.exp(shifted)
        softmax_vals = exp_vals / np.sum(exp_vals, axis=dim, keepdims=True)
        
        out = Tensor(softmax_vals,_children=(self,))
        
        def _SOFTMAX_backward():
            # 正确处理softmax梯度
            # 梯度公式: ∂softmax(x)_i/∂x_j = softmax(x)_i * (δ_ij - softmax(x)_j)
            
            # 先计算 softmax * grad
            s_times_grad = softmax_vals * out.grad
            
            # 然后计算 sum(softmax * grad) 沿着softmax的dim维度
            sum_s_times_grad = np.sum(s_times_grad, axis=dim, keepdims=True)
            
            # 最终梯度: softmax * (grad - sum(softmax * grad))
            self.grad += self._handle_broadcast_grad(softmax_vals * (out.grad - sum_s_times_grad), self.shape)
        
        out._backward = _SOFTMAX_backward
        return out
    
    
    def transpose(self):
        self.data = self.data.T
        self.shape = self.data.shape
        return self
    
    @classmethod
    def randn(cls, shape):
        return cls(np.random.randn(*shape))
    
    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape))
    
    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape))
    
    def backward(self):
        queue = [self]
        self.grad = np.ones(self.shape)
        while queue:
            node = queue.pop()
            if node._backward is None:
                continue
            node._backward()
            queue.extend(node._prev)