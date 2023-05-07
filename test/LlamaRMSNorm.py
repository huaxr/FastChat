import torch
from llama_layers import LlamaRMSNorm

# 定义输入张量，batch_size=2，序列长度seq_len=3，特征数hidden_size=4
x = torch.randn(2, 3, 4)

# 创建LlamaRMSNorm模块，输入参数为hidden_size和eps（默认值为1e-8）
rms_norm = LlamaRMSNorm(hidden_size=4, eps=1e-6)

# 对输入张量进行归一化操作
y = rms_norm(x)

# 输出归一化后的张量
print(y)
