from tinygrad.tensor import Tensor
from tinygrad.loss import cross_entropy
import numpy as np

# 创建一个简单的测试用例
print("===== 测试梯度传播 =====")

# 设置随机种子以便结果可重现
np.random.seed(42)

# 创建一个简单的线性层模型
weights = Tensor(np.random.randn(3, 2))
bias = Tensor(np.zeros(2))
print(f"初始权重:\n{weights.data}")

# 创建输入
x = Tensor(np.random.randn(5, 3))
print(f"输入:\n{x.data}")

# 前向传播
logits = x @ weights + bias
print(f"线性输出:\n{logits.data}")

# 应用softmax
probs = logits.softmax()
print(f"Softmax输出:\n{probs.data}")

# 创建目标标签 (类别索引)
targets = Tensor(np.array([0, 1, 0, 1, 0]))
print(f"目标标签:\n{targets.data}")

# 计算损失
loss = cross_entropy(probs, targets)
print(f"损失值: {loss.data}")

# 反向传播
print("开始反向传播...")
loss.backward()

# 检查梯度
print(f"权重梯度:\n{weights.grad}")
print(f"偏置梯度:\n{bias.grad}")

# 检查输入梯度
print(f"输入梯度:\n{x.grad}")

# 验证梯度是否是数值计算的结果
print("\n===== 验证梯度 =====")
# 小扰动测试 (数值梯度检查)
epsilon = 1e-4
original_loss = loss.data

# 检查权重的第一个参数
weights.data[0, 0] += epsilon
logits_plus = x @ weights + bias
probs_plus = logits_plus.softmax()
loss_plus = cross_entropy(probs_plus, targets)
numerical_grad = (loss_plus.data - original_loss) / epsilon
print(f"权重[0,0]的数值梯度: {numerical_grad}")
print(f"权重[0,0]的解析梯度: {weights.grad[0, 0] if weights.grad is not None else None}") 