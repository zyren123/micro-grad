import numpy as np
class GradientFunction:
    def __init__(self, func, tensor=None):
        self.func = func
        self.tensor = tensor  # 保存对应的张量引用
        self.next_functions = []
    
    def add_next_function(self, next_function):
        self.next_functions.append(next_function)
    
    def __call__(self, grad):
        return self.func(grad)
    
    def __repr__(self):
        return f"GradientFunction(func={self.func.__name__})"

class Tensor:
    def __init__(self,data):
        if not isinstance(data, np.ndarray):
            data=np.array(data)
        self.data = data
        self.shape = data.shape
        self.grad_fn = None
        self.grad = None
    
    def __getitem__(self, key):
        return Tensor(self.data[key])
    
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
        return f"Tensor(data={self.data}, grad={self.grad}, grad_fn={self.get_grad_fn()})"

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
        out.grad_fn = GradientFunction(_ADD_backward, out)
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
        out.grad_fn = GradientFunction(_MUL_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        out.grad_fn.add_next_function(other.get_grad_fn())
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        def _POW_backward(grad):
            # 确保在梯度计算中也使用浮点数据
            float_data = self.data.astype(float)
            # 计算 other * data^(other-1) 作为梯度
            power_grad = grad * other * (float_data ** (other - 1))
            return self._handle_broadcast_grad(power_grad, self.shape), None
        
        # 确保数据为浮点型，避免整数负次幂错误
        data = self.data.astype(float)
        out=Tensor(data ** other)
        out.grad_fn = GradientFunction(_POW_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def __matmul__(self, other):
        def _MATMUL_backward(grad):
            # 处理高维张量的矩阵乘法反向传播
            # 对于 C = A @ B，有：
            # dA = grad @ B.T (在最后两个维度上)
            # dB = A.T @ grad (在最后两个维度上)
            
            # 使用swapaxes来交换最后两个维度，相当于转置
            other_T = np.swapaxes(other.data, -2, -1)
            self_T = np.swapaxes(self.data, -2, -1)
            
            grad_self = grad @ other_T
            grad_other = self_T @ grad
            
            return grad_self, grad_other
            
        out=Tensor(self.data @ other.data)
        out.grad_fn = GradientFunction(_MATMUL_backward,out)
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
        """使用现有算子实现ReLU: relu(x) = (x + abs(x)) / 2"""
        return (self + self.abs()) / 2
    
    def exp(self):
        def _EXP_backward(grad):
            return self._handle_broadcast_grad(grad * np.exp(self.data), self.shape)
        out=Tensor(np.exp(self.data))
        out.grad_fn = GradientFunction(_EXP_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def sum(self, axis=None, keepdims=False):
        """使用mean算子实现sum"""
        if axis is None:
            return self.mean() * self.data.size
        else:
            return self.mean(axis=axis, keepdims=keepdims) * self.data.shape[axis]
    
    def mean(self,axis=None,keepdims=False):
        def _MEAN_backward(grad):   
            if axis is None:
                return np.ones_like(self.data) * grad / self.data.size
            else:
                return np.ones_like(self.data) * grad / self.data.shape[axis]
        out=Tensor(np.mean(self.data,axis=axis,keepdims=keepdims))
        out.grad_fn = GradientFunction(_MEAN_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def var(self, axis=None, keepdims=False):
        """计算方差，使用现有算子实现"""
        # var(x) = mean((x - mean(x))^2)
        mean_val = self.mean(axis=axis, keepdims=True)
        diff = self - mean_val
        squared_diff = diff ** 2
        variance = squared_diff.mean(axis=axis, keepdims=keepdims)
        return variance
    
    def log(self):
        def _LOG_backward(grad):
            return self._handle_broadcast_grad(grad / self.data, self.shape)
        out=Tensor(np.log(self.data))
        out.grad_fn = GradientFunction(_LOG_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def abs(self):
        def _ABS_backward(grad):
            return self._handle_broadcast_grad(grad * np.sign(self.data), self.shape)
        out=Tensor(np.abs(self.data))
        out.grad_fn = GradientFunction(_ABS_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def max(self, axis=None, keepdims=False):
        def _MAX_backward(grad):
            # max的梯度只在最大值位置为1，其他位置为0
            max_vals = np.max(self.data, axis=axis, keepdims=True)
            mask = (self.data == max_vals).astype(float)
            if axis is not None and not keepdims:
                # 需要扩展grad的维度以匹配mask
                grad_expanded = np.expand_dims(grad, axis=axis)
                return self._handle_broadcast_grad(mask * grad_expanded, self.shape)
            else:
                return self._handle_broadcast_grad(mask * grad, self.shape)
        out=Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
        out.grad_fn = GradientFunction(_MAX_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def sqrt(self):
        return self ** 0.5
    
    def softmax(self, dim=-1):
        """使用现有算子实现softmax"""
        # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        x_max = self.max(axis=dim, keepdims=True)
        shifted = self - x_max
        exp_shifted = shifted.exp()
        return exp_shifted / exp_shifted.sum(axis=dim, keepdims=True)
    
    def transpose(self, axis1, axis2):
        """张量转置操作
        
        Args:
            *axes: 轴的顺序
                - 如果没有参数：反转所有轴 (等价于 .T)
                - 如果有2个参数：交换指定的两个轴
                - 如果有完整的轴序列：按指定顺序重排轴
        """ 
        out = Tensor(np.swapaxes(self.data, axis1, axis2))
        def _TRANSPOSE_backward(grad):
            # 反向传播时交换相同的轴
            return np.swapaxes(grad, axis1, axis2)     
        out.grad_fn = GradientFunction(_TRANSPOSE_backward, out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def reshape(self, *shape):
        out=Tensor(self.data.reshape(*shape))
        def _RESHAPE_backward(grad):
            return grad.reshape(self.data.shape)
        out.grad_fn = GradientFunction(_RESHAPE_backward,out)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
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

            # The case of _accumulate_grad (叶子节点)
            # if not isinstance(grad_fn, GradientFunction):
            #     grad_fn(current_grad)
            #     continue

            # 为中间节点累积梯度
            if hasattr(grad_fn, 'tensor') and grad_fn.tensor is not None and grad_fn.tensor is not self:
                grad_fn.tensor._accumulate_grad(current_grad)

            grads = grad_fn(current_grad)
            if grads is None:
                continue
            if isinstance(grads, tuple):
                for grad, next_function in zip(grads, grad_fn.next_functions):
                    if grad is not None:
                        task_queue.append((next_function, grad))
            else:
                task_queue.append((grad_fn.next_functions[0], grads))