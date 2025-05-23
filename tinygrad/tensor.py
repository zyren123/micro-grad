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
        self.data = data
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
    
    def __add__(self, other):
        def _ADD_backward(grad):
            # 处理形状不同时的梯度计算（广播情况）
            self_grad = np.copy(grad)
            other_grad = np.copy(grad)
            
            # 如果形状不同，需要对梯度进行求和以匹配原始张量形状
            if self.shape != grad.shape:
                # 按照numpy的广播规则，对多余的维度求和
                axes = tuple(range(len(grad.shape) - len(self.shape)))
                if axes:
                    self_grad = np.sum(self_grad, axis=axes)
                # 对于大小为1的维度进行求和
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        self_grad = np.sum(self_grad, axis=i, keepdims=True)
                        
            if other.shape != grad.shape:
                # 同样处理other的梯度
                axes = tuple(range(len(grad.shape) - len(other.shape)))
                if axes:
                    other_grad = np.sum(other_grad, axis=axes)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        other_grad = np.sum(other_grad, axis=i, keepdims=True)
            
            return self_grad, other_grad
            
        out=Tensor(self.data + other.data)
        out.grad_fn = GradientFunction(_ADD_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
        return out
    
    def __mul__(self, other):
        def _MUL_backward(grad):
            return grad * other.data, grad * self.data
        out=Tensor(self.data * other.data)
        out.grad_fn = GradientFunction(_MUL_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        def _POW_backward(grad):
            return grad * other * self.data ** (other - 1), grad * self.data ** other * np.log(self.data)
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

    def relu(self):
        def _RELU_backward(grad):
            return grad * (self.data > 0)
        out=Tensor(np.maximum(0, self.data))
        out.grad_fn = GradientFunction(_RELU_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def exp(self):
        def _EXP_backward(grad):
            return grad * np.exp(self.data)
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
                    task_queue.append((next_function, grad))
            else:
                task_queue.append((grad_fn.next_functions[0], grads))
    

        
        