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
        if not isinstance(data, np.ndarray):
            data=np.array(data)
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
            # 确保在梯度计算中也使用浮点数据
            float_data = self.data.astype(float)
            # 计算 other * data^(other-1) 作为梯度
            power_grad = grad * other * (float_data ** (other - 1))
            return self._handle_broadcast_grad(power_grad, self.shape), None
        
        # 确保数据为浮点型，避免整数负次幂错误
        data = self.data.astype(float)
        out=Tensor(data ** other)
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
    
    def sum(self, axis=None, keepdims=False):
        """沿指定轴求和"""
        def _SUM_backward(grad):
            if axis is None:
                return np.ones_like(self.data) * grad
            else:
                # 需要将梯度广播回原始形状
                grad_expanded = np.expand_dims(grad, axis) if not keepdims else grad
                return np.broadcast_to(grad_expanded, self.data.shape)
        
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out.grad_fn = GradientFunction(_SUM_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out
    
    def mean(self, axis=None, keepdims=False):
        """沿指定轴求平均值"""
        def _MEAN_backward(grad):
            if axis is None:
                size = self.data.size
                return np.ones_like(self.data) * grad / size
            else:
                # 计算指定轴的大小
                if isinstance(axis, int):
                    size = self.data.shape[axis]
                else:
                    size = np.prod([self.data.shape[ax] for ax in axis])
                
                grad_expanded = np.expand_dims(grad, axis) if not keepdims else grad
                return np.broadcast_to(grad_expanded, self.data.shape) / size
        
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
        out.grad_fn = GradientFunction(_MEAN_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def log(self):
        def _LOG_backward(grad):
            return self._handle_broadcast_grad(grad / self.data, self.shape)
        out=Tensor(np.log(self.data))
        out.grad_fn = GradientFunction(_LOG_backward)
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

    def max(self, axis=None, keepdims=False):
        """沿指定轴求最大值"""
        def _MAX_backward(grad):
            if axis is None:
                # 全局最大值
                max_mask = (self.data == np.max(self.data)).astype(float)
                return max_mask * grad / np.sum(max_mask)
            else:
                # 沿指定轴的最大值
                max_vals = np.max(self.data, axis=axis, keepdims=True)
                max_mask = (self.data == max_vals).astype(float)
                
                grad_expanded = np.expand_dims(grad, axis) if not keepdims else grad
                grad_broadcasted = np.broadcast_to(grad_expanded, self.data.shape)
                
                # 处理多个相同最大值的情况
                count_mask = np.sum(max_mask, axis=axis, keepdims=True)
                return max_mask * grad_broadcasted / count_mask
        
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
        out.grad_fn = GradientFunction(_MAX_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def pad(self, pad_width, mode='constant', constant_values=0):
        """填充张量"""
        def _PAD_backward(grad):
            # 移除填充，恢复原始形状
            slices = []
            for i, (pad_before, pad_after) in enumerate(pad_width):
                start = pad_before
                end = grad.shape[i] - pad_after if pad_after > 0 else grad.shape[i]
                slices.append(slice(start, end))
            return grad[tuple(slices)]
        
        out = Tensor(np.pad(self.data, pad_width, mode=mode, constant_values=constant_values))
        out.grad_fn = GradientFunction(_PAD_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def flatten(self, start_dim=1):
        """展平张量"""
        original_shape = self.shape
        
        # 计算新的形状
        if start_dim == 0:
            new_shape = (-1,)
        else:
            batch_dims = original_shape[:start_dim]
            flatten_size = np.prod(original_shape[start_dim:])
            new_shape = batch_dims + (flatten_size,)
        
        output = self.data.reshape(new_shape)
        out = Tensor(output)
        
        def _FLATTEN_backward(grad):
            return grad.reshape(original_shape)
        
        out.grad_fn = GradientFunction(_FLATTEN_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def reshape(self, shape):
        """重塑张量形状"""
        original_shape = self.shape
        output = self.data.reshape(shape)
        out = Tensor(output)
        
        def _RESHAPE_backward(grad):
            return grad.reshape(original_shape)
        
        out.grad_fn = GradientFunction(_RESHAPE_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def transpose(self, axes=None):
        """转置张量"""
        original_shape = self.shape
        
        if axes is None:
            # 默认转置（交换最后两个维度）
            if len(self.shape) < 2:
                axes = list(range(len(self.shape)))
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
        
        output = np.transpose(self.data, axes)
        out = Tensor(output)
        
        def _TRANSPOSE_backward(grad):
            # 反向转置
            inverse_axes = [0] * len(axes)
            for i, ax in enumerate(axes):
                inverse_axes[ax] = i
            return np.transpose(grad, inverse_axes)
        
        out.grad_fn = GradientFunction(_TRANSPOSE_backward)
        out.grad_fn.add_next_function(self.get_grad_fn())
        return out

    def __getitem__(self, key):
        """支持索引和切片操作"""
        def _GETITEM_backward(grad):
            result = np.zeros_like(self.data)
            result[key] = grad
            return result
        
        out = Tensor(self.data[key])
        out.grad_fn = GradientFunction(_GETITEM_backward)
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