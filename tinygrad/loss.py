import numpy as np
from tensor import Tensor

def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """计算交叉熵损失
    
    参数:
        y_pred: 预测概率，形状为 (batch_size, num_classes)
        y_true: 真实标签，可以是独热编码 (batch_size, num_classes) 或类别索引 (batch_size,)
        
    返回:
        交叉熵损失
    """
    # 确定y_true是否为类别索引
    if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
        # 对于类别索引形式的标签，我们需要转换为独热编码
        batch_size = y_pred.shape[0]
        num_classes = y_pred.shape[1]
        
        # 提取索引
        indices = y_true.data.astype(np.int32).flatten()
        
        # 创建独热编码
        one_hot = np.zeros((batch_size, num_classes))
        for i in range(batch_size):
            if not (0 <= indices[i] < num_classes):
                raise ValueError(f"标签索引 {indices[i]} 超出了有效范围 [0, {num_classes-1}]")
            one_hot[i, indices[i]] = 1
            
        # 转换为Tensor并使用独热编码计算交叉熵
        one_hot_tensor = Tensor(one_hot)
        return _cross_entropy_one_hot(y_pred, one_hot_tensor)
    else:
        # 已经是独热编码的情况
        return _cross_entropy_one_hot(y_pred, y_true)

def _cross_entropy_one_hot(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """使用独热编码标签计算交叉熵
    
    参数:
        y_pred: 预测概率，形状为 (batch_size, num_classes)
        y_true: 独热编码标签，形状为 (batch_size, num_classes)
        
    返回:
        交叉熵损失
    """
    # 添加一个小值以避免log(0)
    epsilon = 1e-15
    # 把所有小于epsilon的值都替换为epsilon
    clipped_pred = y_pred + epsilon  
    
    # 计算负对数似然
    log_probs = clipped_pred.log()
    total_loss = -1 * (y_true * log_probs).sum()
    
    # 使用浮点数字面量而不是创建新Tensor，避免计算图断开
    # 注意：直接使用标量除法保持梯度连续性
    batch_size = float(y_pred.shape[0])
    loss = total_loss * (1.0 / batch_size)
    
    return loss

def mse(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """计算均方误差
    
    参数:
        y_pred: 预测值
        y_true: 真实值
        
    返回:
        均方误差
    """
    return ((y_pred - y_true) ** 2).mean()


