import torch
import torch.nn as nn

# 创建一个词汇表大小为10000，嵌入维度为300的嵌入层
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)

# 输入一个大小为(3, 5)的LongTensor，其中3表示batch_size，5表示序列长度
input = torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

# 对输入进行嵌入操作，得到大小为(3, 5, 300)的嵌入向量
embedded = embedding(input)

# 输出嵌入向量的形状
print(embedded.shape)  # torch.Size([3, 5, 300])
