import numpy as np
from tensor import Tensor

def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return -Tensor.sum(y_true * Tensor.log(y_pred))


def mse(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return Tensor.mean((y_pred - y_true) ** 2)


