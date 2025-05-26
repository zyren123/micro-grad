#!/usr/bin/env python3
"""
使用CNN训练MNIST数据集
"""

import numpy as np
import time
from tensor import Tensor
from module import CNN, Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
from loss import cross_entropy

class SimpleOptimizer:
    """简单的SGD优化器"""
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr
    
    def step(self):
        """执行一步优化"""
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """清零梯度"""
        for param in self.params:
            param.grad = None

def generate_mnist_like_data(num_samples=1000, num_classes=10):
    """生成类似MNIST的模拟数据"""
    print(f"生成 {num_samples} 个模拟MNIST样本...")
    
    # 生成随机图像数据 (28x28)
    X = np.random.randn(num_samples, 1, 28, 28) * 0.5
    
    # 为每个类别添加一些特征模式
    y = np.random.randint(0, num_classes, num_samples)
    
    for i in range(num_samples):
        label = y[i]
        # 在图像中添加一些与标签相关的模式
        # 这里简单地在不同位置添加亮点
        row = (label // 3) * 8 + 5
        col = (label % 3) * 8 + 5
        X[i, 0, row:row+3, col:col+3] += 2.0
        
        # 添加一些噪声
        X[i] += np.random.randn(1, 28, 28) * 0.1
    
    return X, y

def create_simple_cnn():
    """创建一个简单的CNN模型"""
    model = Sequential(
        # 第一个卷积块
        Conv2d(1, 16, kernel_size=5, padding=2),  # 28x28 -> 28x28
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),       # 28x28 -> 14x14
        
        # 第二个卷积块
        Conv2d(16, 32, kernel_size=5, padding=2), # 14x14 -> 14x14
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2),       # 14x14 -> 7x7
        
        # 分类器
        Flatten(),
        Linear(32 * 7 * 7, 128),
        ReLU(),
        Linear(128, 10)
    )
    return model

def train_epoch(model, optimizer, X_train, y_train, batch_size=32):
    """训练一个epoch"""
    num_samples = len(X_train)
    total_loss = 0.0
    correct = 0
    num_batches = 0
    
    # 随机打乱数据
    indices = np.random.permutation(num_samples)
    
    for i in range(0, num_samples, batch_size):
        # 获取批次数据
        batch_indices = indices[i:i+batch_size]
        batch_X = X_train[batch_indices]
        batch_y = y_train[batch_indices]
        
        # 转换为Tensor
        x = Tensor(batch_X)
        labels = Tensor(batch_y)
        
        # 前向传播
        output = model(x)
        probs = output.softmax()
        
        # 计算损失
        loss = cross_entropy(probs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.data
        predictions = np.argmax(probs.data, axis=1)
        correct += np.sum(predictions == batch_y)
        num_batches += 1
        
        # 打印进度
        if num_batches % 10 == 0:
            print(f"  批次 {num_batches}, 损失: {loss.data:.4f}")
    
    avg_loss = total_loss / num_batches
    accuracy = correct / num_samples
    return avg_loss, accuracy

def evaluate(model, X_test, y_test, batch_size=32):
    """评估模型"""
    num_samples = len(X_test)
    total_loss = 0.0
    correct = 0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_X = X_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        
        # 转换为Tensor
        x = Tensor(batch_X)
        labels = Tensor(batch_y)
        
        # 前向传播（无梯度计算）
        output = model(x)
        probs = output.softmax()
        
        # 计算损失
        loss = cross_entropy(probs, labels)
        
        # 统计
        total_loss += loss.data
        predictions = np.argmax(probs.data, axis=1)
        correct += np.sum(predictions == batch_y)
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = correct / num_samples
    return avg_loss, accuracy

def main():
    """主训练函数"""
    print("🚀 开始CNN训练示例")
    print("=" * 50)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成数据
    print("📊 准备数据...")
    X_train, y_train = generate_mnist_like_data(num_samples=800, num_classes=10)
    X_test, y_test = generate_mnist_like_data(num_samples=200, num_classes=10)
    
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    
    # 创建模型
    print("\n🏗️  创建模型...")
    model = create_simple_cnn()
    print(f"模型结构:\n{model}")
    print(f"总参数数量: {model.total_params():,}")
    
    # 创建优化器
    optimizer = SimpleOptimizer(model.params, lr=0.001)
    
    # 训练参数
    num_epochs = 5
    batch_size = 16
    
    print(f"\n🎯 开始训练...")
    print(f"训练轮数: {num_epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {optimizer.lr}")
    print("-" * 50)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, optimizer, X_train, y_train, batch_size)
        
        # 评估
        test_loss, test_acc = evaluate(model, X_test, y_test, batch_size)
        
        epoch_time = time.time() - start_time
        
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
        print(f"耗时: {epoch_time:.2f}秒")
    
    print("\n🎉 训练完成！")
    
    # 最终评估
    print("\n📈 最终结果:")
    final_test_loss, final_test_acc = evaluate(model, X_test, y_test, batch_size)
    print(f"最终测试准确率: {final_test_acc:.4f}")
    
    # 显示一些预测示例
    print("\n🔍 预测示例:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        x = Tensor(X_test[idx:idx+1])
        output = model(x)
        probs = output.softmax()
        predicted = np.argmax(probs.data)
        actual = y_test[idx]
        confidence = probs.data[0, predicted]
        
        print(f"样本 {idx}: 预测={predicted}, 实际={actual}, 置信度={confidence:.3f} {'✓' if predicted == actual else '✗'}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"❌ 训练过程中出现错误: {type(e).__name__} - {e}")
        traceback.print_exc() 