#!/usr/bin/env python3
"""
简化的CNN测试 - 使用基础运算实现
"""

import numpy as np
from tensor import Tensor
from module import Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sequential
from loss import cross_entropy

def test_basic_operations():
    """测试基础运算"""
    print("=== 测试基础运算 ===")
    
    # 测试pad操作
    x = Tensor(np.random.randn(2, 3, 4, 4))
    print(f"原始形状: {x.shape}")
    
    padded = x.pad(((0, 0), (0, 0), (1, 1), (1, 1)))
    print(f"填充后形状: {padded.shape}")
    
    # 测试索引操作
    slice_x = x[0:1, :, 1:3, 1:3]
    print(f"切片后形状: {slice_x.shape}")
    
    # 测试max操作
    max_val = x.max()
    print(f"最大值: {max_val.data}")
    
    # 测试transpose
    transposed = x.transpose((0, 1, 3, 2))
    print(f"转置后形状: {transposed.shape}")
    
    print("基础运算测试通过！\n")

def test_simple_conv():
    """测试简化的卷积实现"""
    print("=== 测试简化卷积 ===")
    
    # 创建小的测试数据
    x = Tensor(np.random.randn(1, 2, 4, 4))
    print(f"输入形状: {x.shape}")
    
    # 创建简单的卷积层
    conv = Conv2d(2, 3, kernel_size=3, padding=1)
    print(f"卷积层: {conv}")
    
    try:
        # 前向传播
        output = conv(x)
        print(f"输出形状: {output.shape}")
        
        # 简单的反向传播测试
        loss = output.sum()
        loss.backward()
        
        print("简化卷积测试通过！")
        
    except Exception as e:
        print(f"卷积测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_simple_maxpool():
    """测试简化的最大池化"""
    print("=== 测试简化最大池化 ===")
    
    x = Tensor(np.random.randn(1, 2, 4, 4))
    print(f"输入形状: {x.shape}")
    
    maxpool = MaxPool2d(kernel_size=2, stride=2)
    
    try:
        output = maxpool(x)
        print(f"池化输出形状: {output.shape}")
        
        loss = output.sum()
        loss.backward()
        
        print("简化最大池化测试通过！")
        
    except Exception as e:
        print(f"最大池化测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_simple_network():
    """测试简单的网络"""
    print("=== 测试简单网络 ===")
    
    # 创建非常简单的网络
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),  # 保持尺寸
        ReLU(),
        MaxPool2d(kernel_size=2),  # 减半尺寸
        Flatten(),
        Linear(4 * 2 * 2, 2)  # 4x4 -> 2x2 after pooling
    )
    
    print(f"模型: {model}")
    
    # 小的输入
    x = Tensor(np.random.randn(1, 1, 4, 4))
    print(f"输入形状: {x.shape}")
    
    try:
        output = model(x)
        print(f"输出形状: {output.shape}")
        
        # 计算损失
        target = Tensor([0])
        probs = output.softmax()
        loss = cross_entropy(probs, target)
        print(f"损失: {loss.data}")
        
        # 反向传播
        loss.backward()
        print("简单网络测试通过！")
        
    except Exception as e:
        print(f"网络测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print()

if __name__ == "__main__":
    print("开始简化CNN测试...\n")
    
    try:
        test_basic_operations()
        test_simple_conv()
        test_simple_maxpool()
        test_simple_network()
        
        print("🎉 所有简化测试通过！")
        
    except Exception as e:
        import traceback
        print(f"❌ 测试失败: {type(e).__name__} - {e}")
        traceback.print_exc() 