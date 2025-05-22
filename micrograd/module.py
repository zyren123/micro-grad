import random
from tensor import Tensor

class Module:
    def __init__(self):
        self.params = []
        
    def zero_grad(self):
        for p in self.params:
            p.grad = 0
            
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        

class Neuron(Module):
    def __init__(self, nin: int):
        super().__init__()
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(0)
        self.params = self.w + [self.b]
        
    def forward(self, x: list[Tensor]) -> Tensor:
        return sum((wi*xi for wi,xi in zip(self.w, x)), self.b).relu()
    
    def __repr__(self):
        return f"Neuron(nin={len(self.w)})"

class Layer(Module):
    def __init__(self, nin: int, nout: int):
        super().__init__()
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.params = [n.params for n in self.neurons]
        
    def forward(self, x: list[Tensor]) -> list[Tensor]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def __repr__(self):
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)})"
        
class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.params = [p for layer in self.layers for p in layer.params]
        
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f"MLP(nin={len(self.layers[0].neurons[0].w)}, nouts={len(self.layers[-1].neurons)})"


        
        
        