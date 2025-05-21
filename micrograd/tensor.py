class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._op = _op
        self.label = label
        
    def __add__(self, other):
        return Tensor(self.data + other.data, _children=(self, other), _op='+')
    
    def __mul__(self, other):
        return Tensor(self.data * other.data, _children=(self, other), _op='*')
    
    
    