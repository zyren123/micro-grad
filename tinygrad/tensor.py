import numpy as np
class GradientFunction:
    def __init__(self, func):
        self.func = func
        self.next_functions = []
    
    def add_next_function(self, next_function):
        self.next_functions.append(next_function)
    
    def __call__(self, grad):
        return self.func(grad)
    
    def __repr__(self):
        return f"GradientFunction(func={self.func.__name__})"

class Tensor:
    def __init__(self,data):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.shape = data.shape
        self.grad_fn = None
        self.grad = None
    
    def _accumulate_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
    def get_grad_fn(self):
        if self.grad_fn is None:
            return GradientFunction(self._accumulate_grad)
        return self.grad_fn
    
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
        def _ADD_backward(grad):
            self_grad = self._handle_broadcast_grad(grad, self.shape)
            other_grad = self._handle_broadcast_grad(grad, other.shape)
            return self_grad, other_grad

        out=Tensor(self.data + other.data)
        out.grad_fn = GradientFunction(_ADD_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other=Tensor(other)
        def _MUL_backward(grad):
            self_grad = self._handle_broadcast_grad(grad * other.data, self.shape)
            other_grad = self._handle_broadcast_grad(grad * self.data, other.shape)
            return self_grad, other_grad
        out=Tensor(self.data * other.data)
        out.grad_fn = GradientFunction(_MUL_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        def _POW_backward(grad):
            return self._handle_broadcast_grad(grad * other * self.data ** (other - 1), self.shape), None
        out=Tensor(self.data ** other)
        out.grad_fn = GradientFunction(_POW_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def __matmul__(self, other):
        def _MATMUL_backward(grad):
            return grad @ other.data.T, self.data.T @ grad
        out=Tensor(self.data @ other.data)
        out.grad_fn = GradientFunction(_MATMUL_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
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
        def _RELU_backward(grad):
            return self._handle_broadcast_grad(grad * (self.data > 0), self.shape)
        out=Tensor(np.maximum(0, self.data))
        out.grad_fn = GradientFunction(_RELU_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def exp(self):
        def _EXP_backward(grad):
            return self._handle_broadcast_grad(grad * np.exp(self.data), self.shape)
        out=Tensor(np.exp(self.data))
        out.grad_fn = GradientFunction(_EXP_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def sum(self):
        def _SUM_backward(grad):
            return np.ones_like(self.data) * grad
        out=Tensor(np.sum(self.data))
        out.grad_fn = GradientFunction(_SUM_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def mean(self):
        def _MEAN_backward(grad):
            return np.ones_like(self.data) * grad / self.data.size
        out=Tensor(np.mean(self.data))
        out.grad_fn = GradientFunction(_MEAN_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
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
        
        out = Tensor(softmax_vals)
        
        def _SOFTMAX_backward(grad):
            # 正确处理softmax梯度
            # 梯度公式: ∂softmax(x)_i/∂x_j = softmax(x)_i * (δ_ij - softmax(x)_j)
            
            # 先计算 softmax * grad
            s_times_grad = softmax_vals * grad
            
            # 然后计算 sum(softmax * grad) 沿着softmax的dim维度
            sum_s_times_grad = np.sum(s_times_grad, axis=dim, keepdims=True)
            
            # 最终梯度: softmax * (grad - sum(softmax * grad))
            return self._handle_broadcast_grad(softmax_vals * (grad - sum_s_times_grad), self.shape)
        
        out.grad_fn = GradientFunction(_SOFTMAX_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
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
        assert self.grad_fn is not None, "Tensor has no gradient function"
        if self.grad is None:
            self.grad = np.ones(self.shape)

        task_queue = [(self.grad_fn, self.grad)]

        while task_queue:
            grad_fn, current_grad = task_queue.pop()

            # The case of _accumulate_grad
            if not isinstance(grad_fn, GradientFunction):
                grad_fn(current_grad)
                continue

            grads = grad_fn(current_grad)
            if grads is None:
                continue
            if isinstance(grads, tuple):
                for grad, next_function in zip(grads, grad_fn.next_functions):
                    if grad is not None:
                        task_queue.append((next_function, grad))
            else:
                task_queue.append((grad_fn.next_functions[0], grads))