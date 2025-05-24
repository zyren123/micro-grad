import numpy as np
from tensor import Tensor
from module import MLP
from loss import cross_entropy
import matplotlib.pyplot as plt
import time
import torchvision
import torch
import pickle
import os
from sklearn.metrics import classification_report

# 设置随机种子以便结果可重现
np.random.seed(42)
torch.manual_seed(42)

# 创建目录用于保存模型和结果
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 超参数
batch_size = 64
learning_rate = 0.01
epochs = 10
hidden_sizes = [128, 64]  # 隐藏层大小

# 加载MNIST数据集
print("正在加载MNIST数据集...")
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         transform=torchvision.transforms.ToTensor(), 
                                         download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                        transform=torchvision.transforms.ToTensor())

# 获取小批量数据
def get_batch(dataset, batch_size=batch_size):
    indices = torch.randperm(len(dataset))[:batch_size]
    batch_x = []
    batch_y = []
    for i in indices:
        x, y = dataset[i]
        batch_x.append(x.numpy().flatten())
        batch_y.append(y)
    
    # 转换为numpy数组
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    
    # 转换为我们的Tensor类
    return Tensor(batch_x), Tensor(batch_y)

# 创建MLP模型
print("创建模型...")
model = MLP(784, [*hidden_sizes, 10], init_method='he')
print(f"模型总参数量: {model.total_params()}")

# 简单的SGD优化器
def sgd_update(params, lr):
    for param in params:
        if param.grad is not None:
            param.data -= lr * param.grad

# 转换为独热编码
def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    for i, label in enumerate(y.data.astype(int)):
        one_hot[i, label] = 1
    return Tensor(one_hot)

# 评估函数
def evaluate(model, dataset, num_batches=10):
    correct = 0
    total = 0
    
    for i in range(num_batches):
        # 获取批次数据
        x_batch, y_batch = get_batch(dataset, batch_size)
        
        # 前向传播
        output = model(x_batch)
        
        # 获取预测类别
        predictions = np.argmax(output.data, axis=1)
        true_labels = y_batch.data.astype(int)
        
        # 计算准确率
        correct += np.sum(predictions == true_labels)
        total += len(true_labels)
    
    return correct / total

def classification(model, dataset):
    y_true=[]
    y_pred=[]
    for i in range(len(dataset)):
        x, y = dataset[i]
        x_tensor = Tensor(x.numpy().flatten())
        output = model(x_tensor)
        y_true.append(y)
        y_pred.append(np.argmax(output.data))
    return classification_report(y_true, y_pred)

# 可视化测试集上的一些预测结果
def visualize_predictions(model, dataset, num_samples=10):
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(15, 2))
    for i, idx in enumerate(indices):
        # 获取样本
        x, y_true = dataset[idx]
        x_tensor = Tensor(x.numpy().flatten())
        
        # 预测
        output = model(x_tensor)
        prediction = np.argmax(output.data)
        
        # 显示图像和预测结果
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x.numpy()[0], cmap='gray')
        plt.title(f"pred: {prediction}\n true: {y_true}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/predictions.png')
    plt.show()
if __name__ == "__main__":
    # 训练模型
    print("开始训练...")
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # 训练循环
        for iteration in range(len(train_dataset) // batch_size):
            # 获取小批量数据
            x_batch, y_batch = get_batch(train_dataset, batch_size)
            
            # 前向传播
            output = model(x_batch)
            output = output.softmax()
            
            # 计算损失
            y_one_hot = to_one_hot(y_batch)
            loss = cross_entropy(output, y_one_hot)
            epoch_losses.append(loss.data)
            
            # 梯度清零
            model.zero_grad()
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            sgd_update(model.params, learning_rate)
            
            # 打印进度
            if iteration % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Iteration {iteration}/{len(train_dataset) // batch_size}, Loss: {loss.data:.4f}")
        
        # 计算平均训练损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 评估训练集准确率
        train_accuracy = evaluate(model, train_dataset)
        train_accuracies.append(train_accuracy)
        
        # 评估测试集准确率
        test_accuracy = evaluate(model, test_dataset)
        test_accuracies.append(test_accuracy)
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            with open("models/best_model.pkl", "wb") as f:
                pickle.dump(model, f)
        
        # 打印训练统计
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} 完成，用时 {epoch_time:.2f}s")
        print(f"训练损失: {avg_loss:.4f}, 训练准确率: {train_accuracy:.4f}, 测试准确率: {test_accuracy:.4f}")
        print("-" * 60)

    print(f"训练完成! 最佳测试准确率: {best_accuracy:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train')
    plt.plot(test_accuracies, label='test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.show()


    # 加载最佳模型并可视化结果
    with open("models/best_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    visualize_predictions(best_model, test_dataset) 
    print(classification(best_model, test_dataset))