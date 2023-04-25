import torch
import torch.nn.functional as F

# 这个self-attention函数的输入是一个张量inputs，形状为[batch_size, seq_length, hidden_size]，
# 其中batch_size表示批次大小，seq_length表示序列长度，hidden_size表示隐藏状态的维度。
# 在self-attention函数的实现中，我们先将输入通过三个线性层变换成query、key和value张量，
# 再根据它们之间的点积计算attention分数，使用softmax函数计算attention权重，最后将权重与value张量相乘，
# 得到输出的context_vector张量和attention_weights张量。

class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        # inputs shape: [batch_size, seq_length, hidden_size]
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        score = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = F.softmax(score, dim=-1)
        context_vector = torch.matmul(attention_weights, value)
        return context_vector, attention_weights

if __name__ == "__main__":
    # batch_size = 32
    # embed_dim = 128
    # num_heads = 8
    # seq_len = 64

    batch_size = 1
    seq_len = 2
    embed_dim = 3
    num_heads = 8

    # 构建一个 batch_size x seq_len x embed_dim 的输入张量
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)

    # 创建 Self-Attention 模型
    self_attn = SelfAttention(embed_dim)

    # 进行 Self-Attention 计算
    output_tensor = self_attn(input_tensor)

    print(output_tensor.shape)  # 输出应为 batch_size x seq_len x embed_dim
