import torch
from micrograd.module import MLP
import torchvision
dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=torchvision.transforms.ToTensor(),download=True)

def get_batch(dataset:torchvision.datasets.MNIST,batch_size:int=32):
    indices=torch.randperm(len(dataset))[:batch_size]
    x=dataset[indices][0]
    y=dataset[indices][1]
    return x,y


