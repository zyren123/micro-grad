#!/usr/bin/env python3
"""
测试CNN实现
"""

import numpy as np
from tensor import Tensor
from module import CNN, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sequential
from loss import cross_entropy

def test_conv2d():
    """测试卷积层"""
    print("=== 测试Conv2d层 ===")
    
    # 创建测试数据 (batch_size=2, channels=3, height=4, width=4)
    x = Tensor(np.random.randn(2, 3, 4, 4))
    print(f"输入形状: {x.shape}")
    
    # 创建卷积层 (3输入通道 -> 8输出通道, 3x3卷积核)
    conv = Conv2d(3, 8, kernel_size=3, padding=1)
    print(f"卷积层: {conv}")
    print(f"权重形状: {conv.weight.shape}")
    print(f"偏置形状: {conv.bias.shape}")
    
    # 前向传播
    output = conv(x)
    print(f"输出形状: {output.shape}")
    
    # 反向传播测试
    loss = output.sum()
    loss.backward()
    
    print(f"权重梯度形状: {conv.weight.grad.shape}")
    print(f"偏置梯度形状: {conv.bias.grad.shape}")
    print("Conv2d测试通过！\n")

def test_pooling():
    """测试池化层"""
    print("=== 测试池化层 ===")
    
    # 创建测试数据
    x = Tensor(np.random.randn(2, 4, 8, 8))
    print(f"输入形状: {x.shape}")
    
    # 测试最大池化
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    max_output = maxpool(x)
    print(f"MaxPool输出形状: {max_output.shape}")
    
    # 反向传播测试
    loss = max_output.sum()
    loss.backward()
    print(f"输入梯度形状: {x.grad.shape}")
    print("池化测试通过！\n")

def test_flatten():
    """测试展平层"""
    print("=== 测试Flatten层 ===")
    
    # 创建测试数据
    x = Tensor(np.random.randn(2, 4, 3, 3))
    print(f"输入形状: {x.shape}")
    
    flatten = Flatten()
    output = flatten(x)
    print(f"展平后形状: {output.shape}")
    
    # 反向传播测试
    loss = output.sum()
    loss.backward()
    print(f"输入梯度形状: {x.grad.shape}")
    print("Flatten测试通过！\n")

def test_sequential():
    """测试Sequential容器"""
    print("=== 测试Sequential容器 ===")
    
    # 创建简单的序列模型
    model = Sequential(
        Conv2d(1, 16, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Conv2d(16, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Flatten(),
        Linear(32 * 7 * 7, 10)
    )
    
    print(f"模型结构:\n{model}")
    print(f"总参数数量: {model.total_params()}")
    
    # 测试前向传播
    x = Tensor(np.random.randn(2, 1, 28, 28))  # 模拟MNIST数据
    print(f"输入形状: {x.shape}")
    
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    # 测试反向传播
    labels = Tensor([0, 1])
    loss = cross_entropy(output.softmax(), labels)
    print(f"损失值: {loss.data}")
    
    loss.backward()
    print("Sequential测试通过！\n")

def test_cnn_model():
    """测试完整的CNN模型"""
    print("=== 测试完整CNN模型 ===")
    
    # 创建CNN模型
    model = CNN(num_classes=10, input_channels=1)
    print(f"CNN模型:\n{model}")
    print(f"总参数数量: {model.total_params()}")
    
    # 创建模拟MNIST数据
    batch_size = 4
    x = Tensor(np.random.randn(batch_size, 1, 28, 28))
    labels = Tensor(np.random.randint(0, 10, batch_size))
    
    print(f"输入形状: {x.shape}")
    print(f"标签: {labels.data}")
    
    # 前向传播
    output = model(x)
    print(f"模型输出形状: {output.shape}")
    
    # 应用softmax
    probs = output.softmax()
    print(f"概率分布形状: {probs.shape}")
    print(f"第一个样本的概率分布: {probs.data[0]}")
    
    # 计算损失
    loss = cross_entropy(probs, labels)
    print(f"交叉熵损失: {loss.data}")
    
    # 反向传播
    print("开始反向传播...")
    loss.backward()
    
    # 检查梯度
    print("检查部分层的梯度:")
    conv1 = model.features.layers[0]  # 第一个卷积层
    print(f"第一个卷积层权重梯度统计:")
    print(f"  形状: {conv1.weight.grad.shape}")
    print(f"  平均值: {np.mean(conv1.weight.grad):.6f}")
    print(f"  标准差: {np.std(conv1.weight.grad):.6f}")
    
    print("CNN模型测试通过！\n")

def test_gradient_flow():
    """测试梯度流动"""
    print("=== 测试梯度流动 ===")
    
    # 创建简单的CNN
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Flatten(),
        Linear(4 * 14 * 14, 2)
    )
    
    # 输入数据
    x = Tensor(np.random.randn(1, 1, 28, 28))
    target = Tensor([1])
    
    # 前向传播
    output = model(x)
    loss = cross_entropy(output.softmax(), target)
    
    print(f"损失值: {loss.data}")
    
    # 反向传播
    loss.backward()
    
    # 检查每层是否都有梯度
    print("检查各层梯度:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'params') and layer.params:
            for j, param in enumerate(layer.params):
                if param.grad is not None:
                    grad_norm = np.linalg.norm(param.grad)
                    print(f"  层{i} 参数{j}: 梯度范数 = {grad_norm:.6f}")
                else:
                    print(f"  层{i} 参数{j}: 无梯度!")
    
    print("梯度流动测试完成！\n")

if __name__ == "__main__":
    print("开始测试CNN实现...\n")
    
    try:
        test_conv2d()
        test_pooling()
        test_flatten()
        test_sequential()
        test_cnn_model()
        test_gradient_flow()
        
        print("🎉 所有测试通过！CNN实现成功！")
        
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {type(e).__name__} - {e}")
        traceback.print_exc() 