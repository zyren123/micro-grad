class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
        self._prev = set(_children)
        
    def __add__(self, other) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other) -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other) -> "Tensor":
        assert isinstance(other, (int, float)), "幂运算只支持整数或浮点数"
        out = Tensor(self.data ** other, _children=(self,), _op='**')
        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def relu(self) -> "Tensor":
        out = Tensor(0 if self.data < 0 else self.data, _children=(self,), _op='relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        queue = [self]
        self.grad = 1.0
        while queue:
            node = queue.pop()
            if node._backward is None:
                continue
            node._backward()
            queue.extend(node._prev)
            
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"