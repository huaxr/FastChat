import torch
from llama_attention import LlamaAttention

# 假设输入向量的维度是 128，self-attention 的头数为 4，每个头的维度为 32
input_dim = 128
num_heads = 4
head_dim = input_dim // num_heads

# 创建 LlamaAttention 模块
attention = LlamaAttention(input_dim=input_dim, num_heads=num_heads, head_dim=head_dim)

# 假设我们有一个输入张量 x，维度为 (batch_size, sequence_length, input_dim)
batch_size = 2
sequence_length = 10
x = torch.randn(batch_size, sequence_length, input_dim)

# 进行 self-attention 操作，得到输出张量 y，维度为 (batch_size, sequence_length, input_dim)
y = attention(x)

# 输出张量的维度为 (batch_size, sequence_length, input_dim)
print(y.shape)
