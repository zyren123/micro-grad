import numpy as np
class GradientFunction:
    def __init__(self, func):
        self.func = func
        self.next_functions = []
    
    def add_next_function(self, next_function):
        self.next_functions.append(next_function)

class Tensor:
    def __init__(self,data):
        self.data = data
        self.shape = data.shape
        self.grad_fn = None
        self.grad = 0
    
    def _accumulate_grad(self, grad):
        # if self.grad is None:
        #     self.grad = grad
        # else:
        #     self.grad += grad
        self.grad += grad
        
    def get_grad_fn(self):
        if self.grad_fn is None:
            return self._accumulate_grad
        return self.grad_fn
    
    def __add__(self, other):
        def _ADD_backward(grad):
            return np.copy(grad), np.copy(grad)
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

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape))
    
    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape))
    
    
        
        