from micrograd.tensor import Tensor
import random
import numpy as np
import matplotlib.pyplot as plt

# 模型参数初始化
w = Tensor(random.uniform(-1, 1))
b = Tensor(random.uniform(-1, 1))

def model(x):
    return x * w + b

# 生成线性回归数据集
def generate_dataset(n_samples=100, true_w=3.5, true_b=2.0, noise=0.5, x_range=(-10, 10)):
    """
    生成线性回归数据集
    
    参数:
        n_samples: 样本数量
        true_w: 真实权重
        true_b: 真实偏置
        noise: 噪声水平
        x_range: x值的范围
        
    返回:
        xs: x值列表
        ys: y值列表
    """
    xs = []
    ys = []
    
    for _ in range(n_samples):
        x = random.uniform(x_range[0], x_range[1])
        # y = wx + b + 噪声
        y = true_w * x + true_b + random.uniform(-noise, noise)
        xs.append(x)
        ys.append(y)
    
    return xs, ys

def mse_loss(y_pred, y_true):
    loss=Tensor(0.0)
    for y_p, y_t in zip(y_pred, y_true):
        loss += (y_p - y_t) ** 2
    return loss / len(y_pred)

def train_model(xs, ys, epochs=100, learning_rate=0.01):
    losses = []
    
    for epoch in range(epochs):
        y_preds = [model(x) for x in xs]
        loss=mse_loss(y_preds, ys)
        losses.append(loss.data)
        
        loss.backward()
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad
        
        w.grad = 0
        b.grad = 0
        
        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss: {loss.data:.4f}, w: {w.data:.4f}, b: {b.data:.4f}")
    
    return losses

# 绘制数据集
def plot_dataset(xs, ys, pred_ys=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, label='data')
    
    if pred_ys is not None:
        # 按x值排序，以便绘制平滑的预测线
        sorted_pairs = sorted(zip(xs, pred_ys))
        sorted_xs, sorted_pred_ys = zip(*sorted_pairs)
        plt.plot(sorted_xs, sorted_pred_ys, 'r-', label='prediction')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_regression_data.png')
    plt.close()

# 绘制损失变化图
def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss')
    plt.grid(True)
    plt.savefig('linear_regression_loss.png')
    plt.close()

# 生成数据集
true_w, true_b = 3.5, 2.0
xs, ys = generate_dataset(true_w=true_w, true_b=true_b)

# 打印数据集信息
print(f"生成了{len(xs)}个数据点")
print(f"真实参数: w={true_w}, b={true_b}")
print(f"初始参数: w={w.data:.4f}, b={b.data:.4f}")

# 绘制原始数据
plot_dataset(xs, ys)

# 训练模型
losses = train_model(xs, ys, epochs=100, learning_rate=0.001)

# 评估模型
pred_ys = [model(Tensor(x)).data for x in xs]
print(f"训练后参数: w={w.data:.4f}, b={b.data:.4f}")

# 绘制训练后的模型预测
plot_dataset(xs, ys, pred_ys)

# 绘制损失变化
plot_loss(losses)


