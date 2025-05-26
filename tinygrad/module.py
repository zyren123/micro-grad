from tensor import Tensor, GradientFunction
# from torch.nn import Linear as TorchLinear
# from torch import tensor as TorchTensor
# import torch
from loss import cross_entropy
import numpy as np

class Module:
    def __init__(self):
        self.params: list[Tensor] = []

    def zero_grad(self):
        for p in self.params:
            p.grad = 0
            
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def total_params(self):
        return sum(p.data.size for p in self.params)
    
class Linear(Module):
    def __init__(self, nin: int, nout: int, init_method='he'):
        """线性层
        
        参数:
            nin: 输入特征数
            nout: 输出特征数
            init_method: 初始化方法，可选'he'、'xavier'或'normal'
        """
        super().__init__()
        
        if init_method == 'he':
            # He初始化 - 适用于ReLU激活函数
            # W ~ N(0, sqrt(2/n_in))
            scale = np.sqrt(2.0 / nin)
            self.w = Tensor(np.random.randn(nin, nout) * scale)
        elif init_method == 'xavier':
            # Xavier/Glorot初始化 - 适用于tanh或sigmoid激活函数
            # W ~ N(0, sqrt(2/(n_in + n_out)))
            scale = np.sqrt(2.0 / (nin + nout))
            self.w = Tensor(np.random.randn(nin, nout) * scale)
        else:
            # 普通随机初始化
            self.w = Tensor.randn((nin, nout))
            
        self.b = Tensor.zeros((nout,))
        self.params = [self.w, self.b]
        
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w + self.b
    
    def __repr__(self):
        return f"Linear(nin={self.w.shape[0]}, nout={self.w.shape[1]})"


class MLP(Module):
    def __init__(self, nin:int, nouts:list[int], init_method='he'):
        super().__init__()
        layers=[]
        for nout in nouts:
            # 使用指定的初始化方法
            layer = Linear(nin, nout, init_method=init_method)
            layers.append(layer)
            self.params.extend(layers[-1].params)
            nin = nout
        self.layers = layers
        
    def forward(self, x:Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 最后一层不使用激活函数
            if i < len(self.layers) - 1:
                x = x.relu() 
        return x
    
    def __repr__(self):
        return f"MLP(layers={self.layers})"


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, init_method='he'):
        """2D卷积层
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长，默认为1
            padding: 填充，默认为0
            init_method: 初始化方法
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 权重初始化
        if init_method == 'he':
            # He初始化适用于ReLU
            fan_in = in_channels * kernel_size * kernel_size
            scale = np.sqrt(2.0 / fan_in)
            self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        elif init_method == 'xavier':
            # Xavier初始化
            fan_in = in_channels * kernel_size * kernel_size
            fan_out = out_channels * kernel_size * kernel_size
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        else:
            self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        
        self.bias = Tensor(np.zeros(out_channels))
        self.params = [self.weight, self.bias]
    
    def forward(self, x: Tensor) -> Tensor:
        """使用基础运算实现卷积"""
        batch_size, in_channels, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            x_padded = x.pad(pad_width, mode='constant', constant_values=0)
        else:
            x_padded = x
        
        # 使用im2col方法实现卷积
        # 将输入展开为矩阵形式
        patches = []
        for b in range(batch_size):
            batch_patches = []
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = ow * self.stride
                    w_end = w_start + self.kernel_size
                    
                    # 提取patch: (in_channels, kernel_size, kernel_size)
                    patch = x_padded[b:b+1, :, h_start:h_end, w_start:w_end]
                    # 展平为 (in_channels * kernel_size * kernel_size,)
                    patch_flat = patch.reshape((in_channels * self.kernel_size * self.kernel_size,))
                    batch_patches.append(patch_flat)
            
            # 堆叠所有patches: (out_h * out_w, in_channels * kernel_size * kernel_size)
            if batch_patches:
                batch_matrix = self._stack_tensors(batch_patches, axis=0)
                patches.append(batch_matrix)
        
        # 堆叠所有batch: (batch_size, out_h * out_w, in_channels * kernel_size * kernel_size)
        if patches:
            input_matrix = self._stack_tensors(patches, axis=0)
        else:
            input_matrix = Tensor(np.zeros((batch_size, out_h * out_w, in_channels * self.kernel_size * self.kernel_size)))
        
        # 重塑权重: (out_channels, in_channels * kernel_size * kernel_size)
        weight_matrix = self.weight.reshape((self.out_channels, in_channels * self.kernel_size * self.kernel_size))
        
        # 矩阵乘法: (batch_size, out_h * out_w, out_channels)
        output_flat = input_matrix @ weight_matrix.transpose()
        
        # 添加偏置
        bias_expanded = self.bias.reshape((1, 1, self.out_channels))
        output_flat = output_flat + bias_expanded
        
        # 重塑为输出形状: (batch_size, out_channels, out_h, out_w)
        output = output_flat.reshape((batch_size, out_h, out_w, self.out_channels))
        output = output.transpose((0, 3, 1, 2))  # 调整维度顺序
        
        return output
    
    def _stack_tensors(self, tensors, axis=0):
        """手动实现tensor堆叠"""
        if not tensors:
            return Tensor(np.array([]))
        
        # 获取所有tensor的数据
        arrays = [t.data for t in tensors]
        stacked_data = np.stack(arrays, axis=axis)
        
        # 创建新的tensor
        result = Tensor(stacked_data)
        
        # 简化的梯度函数 - 实际应该更复杂
        def _STACK_backward(grad):
            # 这里简化处理，实际需要根据axis正确分割梯度
            return [grad[i] for i in range(len(tensors))]
        
        result.grad_fn = GradientFunction(_STACK_backward)
        for t in tensors:
            result.grad_fn.add_next_function(t.get_grad_fn())
        
        return result
    
    def __repr__(self):
        return f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        """2D最大池化层 - 使用基础运算实现"""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            x_padded = x.pad(pad_width, mode='constant', constant_values=-np.inf)
        else:
            x_padded = x
        
        # 收集所有池化窗口的最大值
        output_patches = []
        for b in range(batch_size):
            for c in range(channels):
                channel_patches = []
                for oh in range(out_h):
                    row_patches = []
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取池化窗口
                        window = x_padded[b:b+1, c:c+1, h_start:h_end, w_start:w_end]
                        # 计算最大值
                        max_val = window.max()
                        row_patches.append(max_val)
                    
                    # 堆叠行
                    if row_patches:
                        row_tensor = self._stack_tensors(row_patches, axis=0)
                        channel_patches.append(row_tensor)
                
                # 堆叠通道的所有行
                if channel_patches:
                    channel_tensor = self._stack_tensors(channel_patches, axis=0)
                    output_patches.append(channel_tensor)
        
        # 重塑输出
        if output_patches:
            # 将所有patches重新组织为正确的形状
            output_data = np.zeros((batch_size, channels, out_h, out_w))
            idx = 0
            for b in range(batch_size):
                for c in range(channels):
                    output_data[b, c] = output_patches[idx].data
                    idx += 1
            
            return Tensor(output_data)
        else:
            return Tensor(np.zeros((batch_size, channels, out_h, out_w)))
    
    def _stack_tensors(self, tensors, axis=0):
        """简化的tensor堆叠实现"""
        if not tensors:
            return Tensor(np.array([]))
        
        arrays = [t.data for t in tensors]
        stacked_data = np.stack(arrays, axis=axis)
        return Tensor(stacked_data)
    
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int = None, padding: int = 0):
        """2D平均池化层 - 使用基础运算实现"""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            x_padded = x.pad(pad_width, mode='constant', constant_values=0)
        else:
            x_padded = x
        
        # 收集所有池化窗口的平均值
        output_data = np.zeros((batch_size, channels, out_h, out_w))
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取池化窗口并计算平均值
                        window = x_padded[b:b+1, c:c+1, h_start:h_end, w_start:w_end]
                        avg_val = window.mean()
                        output_data[b, c, oh, ow] = avg_val.data
        
        return Tensor(output_data)
    
    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        """展平层"""
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


class ReLU(Module):
    def __init__(self):
        """ReLU激活函数层"""
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"


class Sequential(Module):
    def __init__(self, *layers):
        """顺序容器，按顺序执行层"""
        super().__init__()
        self.layers = list(layers)
        # 收集所有层的参数
        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.params.extend(layer.params)
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        layer_strs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return f"Sequential(\n" + "\n".join(layer_strs) + "\n)"


class CNN(Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        """简单的CNN模型
        
        Args:
            num_classes: 分类数量
            input_channels: 输入通道数（如MNIST为1，CIFAR-10为3）
        """
        super().__init__()
        
        # 特征提取部分
        self.features = Sequential(
            Conv2d(input_channels, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            
            Conv2d(64, 128, kernel_size=3, padding=1),  # 7x7 -> 7x7
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3 (对于28x28输入)
        )
        
        # 分类器部分
        self.classifier = Sequential(
            Flatten(),
            Linear(128 * 3 * 3, 256),  # 根据特征图大小调整
            ReLU(),
            Linear(256, num_classes)
        )
        
        # 收集所有参数
        self.params.extend(self.features.params)
        self.params.extend(self.classifier.params)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def __repr__(self):
        return f"CNN(\n  features={self.features}\n  classifier={self.classifier}\n)"
        
if __name__ == "__main__":
    try:
        print("创建模型...")
        model = MLP(784, [1024, 10])
        
        print("创建输入和计算前向传播...")
        input = Tensor.randn((2, 784))
        
        # 跟踪中间层输出
        intermediate_outputs = []
        x = input
        for i, layer in enumerate(model.layers):
            x = layer(x)
            intermediate_outputs.append(x)
            print(f"第{i+1}层输出形状: {x.shape}, 平均值: {np.mean(x.data)}, 最大值: {np.max(x.data)}, 最小值: {np.min(x.data)}")
        
        output = intermediate_outputs[-1]
        
        print("应用softmax...")
        output = output.softmax()
        print(f"Softmax输出平均值: {np.mean(output.data)}, 最大值: {np.max(output.data)}, 最小值: {np.min(output.data)}")
        
        print("计算损失...")
        labels = Tensor([0, 1])
        print(f"标签: {labels.data}")
        print(f"输出形状: {output.shape}")
        
        loss = cross_entropy(output, labels)
        print(f"损失值: {loss.data}")
        
        print("开始反向传播...")
        loss.backward()
        
        print("检查各层梯度...")
        for i, layer in enumerate(model.layers):
            w_grad = layer.params[0].grad
            b_grad = layer.params[1].grad
            
            print(f"第{i+1}层权重梯度:")
            print(f"  形状: {w_grad.shape if w_grad is not None else None}")
            if w_grad is not None:
                print(f"  平均值: {np.mean(w_grad)}")
                print(f"  最大值: {np.max(w_grad)}")
                print(f"  最小值: {np.min(w_grad)}")
                print(f"  非零元素比例: {np.count_nonzero(w_grad) / w_grad.size}")
            
            print(f"第{i+1}层偏置梯度:")
            print(f"  形状: {b_grad.shape if b_grad is not None else None}")
            if b_grad is not None:
                print(f"  平均值: {np.mean(b_grad)}")
                print(f"  最大值: {np.max(b_grad)}")
                print(f"  最小值: {np.min(b_grad)}")
                print(f"  非零元素比例: {np.count_nonzero(b_grad) / b_grad.size}")
        
        # 检查输入梯度
        if input.grad is not None:
            print(f"输入梯度:")
            print(f"  形状: {input.grad.shape}")
            print(f"  平均值: {np.mean(input.grad)}")
            print(f"  最大值: {np.max(input.grad)}")
            print(f"  最小值: {np.min(input.grad)}")
            print(f"  非零元素比例: {np.count_nonzero(input.grad) / input.grad.size}")
        
        print("反向传播完成！")
        
    except Exception as e:
        import traceback
        print(f"发生错误: {type(e).__name__} - {e}")
        traceback.print_exc()

    
    # 测试我们自己的实现
    # print("测试我们的实现：")
    # x=Tensor.randn((10,5))
    # linear=Linear(5,10)
    # my_output=linear(x)
    # my_output.backward()
    
    # print("我们的权重梯度形状:", linear.w.data.shape)
    # print("我们的偏置梯度形状:", linear.b.grad.shape)
    # print("我们的权重梯度:")
    # print(linear.w.grad)
    # print("我们的偏置梯度:")
    # print(linear.b.grad)
    
    # print("\n" + "*"*80 + "\n")
    
    # # 测试PyTorch的实现
    # print("测试PyTorch的实现：")
    # torch_linear=TorchLinear(5,10)
    # torch_linear.weight.data=TorchTensor(linear.w.data.transpose())
    # torch_linear.bias.data=TorchTensor(linear.b.data.transpose())
    # torch_output=torch_linear(TorchTensor(x.data))
    # torch_output.backward(gradient=torch.ones_like(torch_output))
    
    # print("PyTorch权重梯度形状:", torch_linear.weight.grad.T.shape)
    # print("PyTorch偏置梯度形状:", torch_linear.bias.grad.shape)
    # print("PyTorch权重梯度:")
    # print(torch_linear.weight.grad.T)
    # print("PyTorch偏置梯度:")
    # print(torch_linear.bias.grad)
        