from tensor import Tensor
from torch.nn import Linear as TorchLinear
from torch import tensor as TorchTensor
import torch
from loss import cross_entropy
import numpy as np

class Module:
    def __init__(self):
        self.params: list[Tensor] = []

    def zero_grad(self):
        for p in self.params:
            p.grad = 0
            
    def __call__(self, x: Tensor,**kwargs) -> Tensor:
        return self.forward(x,**kwargs)
    
    def forward(self, x: Tensor,**kwargs) -> Tensor:
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

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        self.params.extend(layer.params for layer in layers)
        
    def forward(self, x:Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        return f"Sequential(layers={self.layers})"

class Attention(Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q = Linear(d_model, d_model)
        self.k = Linear(d_model, d_model)
        self.v = Linear(d_model, d_model)
        self.o = Linear(d_model, d_model)
        
    def forward(self, x:Tensor, is_causal=False) -> Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d_model // self.n_heads)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d_model // self.n_heads)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d_model // self.n_heads)
        
        # 转置以便进行多头注意力计算: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        attn_scores = q @ k.transpose(-2, -1)
        
        if is_causal:
            # 创建因果mask：下三角为0，上三角为-inf
            seq_len = attn_scores.shape[-1]
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)  # 上三角设为-inf
            attn_scores = attn_scores + mask
        attn_scores = attn_scores / np.sqrt(self.d_model // self.n_heads)
        attn_weights = attn_scores.softmax()
        
        out = attn_weights @ v
        
        # 转置回原来的形状并reshape
        out = out.transpose(1,2)
        out = out.reshape(out.shape[0], out.shape[1], self.d_model)
        
        return self.o(out)

def test_attention():
    x=Tensor.randn((1,10,12))
    print(f"输入 x 的形状: {x.shape}")
    attn=Attention(12,4)
    attn_weights=attn(x,is_causal=True)
    attn_weights.backward()
    print(attn.q.w.grad)
    print(attn.k.w.grad)
    print(attn.v.w.grad)
    print(attn.o.w.grad)
    print(x.grad)
    
def compare_attention_with_pytorch():
    """比较我们的注意力机制实现和PyTorch的实现"""
    print("=" * 60)
    print("比较自定义注意力机制和PyTorch实现")
    print("=" * 60)
    
    # 设置相同的随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 参数设置
    batch_size, seq_len, d_model, n_heads = 1, 8, 12, 4
    head_dim = d_model // n_heads
    
    # 创建相同的输入数据
    input_data = np.random.randn(batch_size, seq_len, d_model)
    
    # 我们的实现
    print("测试我们的实现:")
    our_x = Tensor(input_data.copy())
    our_attn = Attention(d_model, n_heads)
    
    # PyTorch实现
    print("测试PyTorch实现:")
    torch_x = torch.tensor(input_data.copy(), requires_grad=True, dtype=torch.float32)
    torch_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    # 将我们的权重复制到PyTorch模型中
    with torch.no_grad():
        # 注意PyTorch的权重形状是 (d_model, d_model)，我们的是 (d_model, d_model)
        # 确保所有数据都是float32类型
        torch_attn.in_proj_weight[:d_model] = torch.tensor(our_attn.q.w.data.T, dtype=torch.float32)  # Q权重
        torch_attn.in_proj_weight[d_model:2*d_model] = torch.tensor(our_attn.k.w.data.T, dtype=torch.float32)  # K权重  
        torch_attn.in_proj_weight[2*d_model:] = torch.tensor(our_attn.v.w.data.T, dtype=torch.float32)  # V权重
        
        torch_attn.in_proj_bias[:d_model] = torch.tensor(our_attn.q.b.data, dtype=torch.float32)  # Q偏置
        torch_attn.in_proj_bias[d_model:2*d_model] = torch.tensor(our_attn.k.b.data, dtype=torch.float32)  # K偏置
        torch_attn.in_proj_bias[2*d_model:] = torch.tensor(our_attn.v.b.data, dtype=torch.float32)  # V偏置
        
        torch_attn.out_proj.weight.data = torch.tensor(our_attn.o.w.data.T, dtype=torch.float32)
        torch_attn.out_proj.bias.data = torch.tensor(our_attn.o.b.data, dtype=torch.float32)
    
    # 前向传播
    print("\n前向传播比较:")
    
    # 我们的前向传播
    our_output = our_attn(our_x, is_causal=False)  # 先测试非因果注意力
    print(f"我们的输出形状: {our_output.shape}")
    print(f"我们的输出前5个元素: {our_output.data.flatten()[:5]}")
    
    # PyTorch前向传播
    torch_output, _ = torch_attn(torch_x, torch_x, torch_x)
    print(f"PyTorch输出形状: {torch_output.shape}")
    print(f"PyTorch输出前5个元素: {torch_output.detach().flatten()[:5].tolist()}")
    
    # 比较输出差异
    torch_output_np = torch_output.detach().cpu().numpy()
    output_diff = np.abs(our_output.data - torch_output_np)
    print(f"输出最大差异: {np.max(output_diff)}")
    print(f"输出平均差异: {np.mean(output_diff)}")
    
    # 反向传播比较
    print("\n反向传播比较:")
    
    # 创建相同的损失
    our_loss = our_output.sum()
    torch_loss = torch_output.sum()
    
    # 反向传播
    our_loss.backward()
    torch_loss.backward()
    
    # 比较输入梯度
    torch_x_grad_np = torch_x.grad.detach().cpu().numpy()
    input_grad_diff = np.abs(our_x.grad - torch_x_grad_np)
    print(f"输入梯度最大差异: {np.max(input_grad_diff)}")
    print(f"输入梯度平均差异: {np.mean(input_grad_diff)}")
    
    # 比较权重梯度
    # Q权重梯度
    q_grad_torch = torch_attn.in_proj_weight.grad[:d_model].T.detach().cpu().numpy()
    q_grad_diff = np.abs(our_attn.q.w.grad - q_grad_torch)
    print(f"Q权重梯度最大差异: {np.max(q_grad_diff)}")
    
    # K权重梯度  
    k_grad_torch = torch_attn.in_proj_weight.grad[d_model:2*d_model].T.detach().cpu().numpy()
    k_grad_diff = np.abs(our_attn.k.w.grad - k_grad_torch)
    print(f"K权重梯度最大差异: {np.max(k_grad_diff)}")
    
    # V权重梯度
    v_grad_torch = torch_attn.in_proj_weight.grad[2*d_model:].T.detach().cpu().numpy()
    v_grad_diff = np.abs(our_attn.v.w.grad - v_grad_torch)
    print(f"V权重梯度最大差异: {np.max(v_grad_diff)}")
    
    # 输出层权重梯度
    o_grad_torch = torch_attn.out_proj.weight.grad.T.detach().cpu().numpy()
    o_grad_diff = np.abs(our_attn.o.w.grad - o_grad_torch)
    print(f"输出权重梯度最大差异: {np.max(o_grad_diff)}")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        # 首先比较我们的注意力机制和PyTorch的实现
        compare_attention_with_pytorch()
        
        test_attention()
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
        