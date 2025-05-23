from tensor import Tensor
import torch
if __name__ == "__main__":
    logits=[Tensor(2.0), Tensor(1.0), Tensor(0.1)]
    o=Tensor.softmax(logits)
    o=sum(o)
    o.backward()
    print(logits)
    
    torch_logits=torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    torch_o=torch.softmax(torch_logits, dim=0)
    torch_o=torch.sum(torch_o)
    torch_o.backward()
    print(torch_logits.grad)