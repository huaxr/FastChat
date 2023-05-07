import torch
import torch.nn as nn

# 在自然语言处理任务中，常常使用词向量（word embedding）表示每个单词，将每个单词映射到一个固定长度的实数向量，以便进行进一步的处理。这些词向量可以通过不同的方式获得，其中一种方式是使用预训练的词向量。这些词向量通常是在大型语料库上通过无监督学习的方式训练得到的，例如 Word2Vec 和 GloVe 等算法。预训练的词向量已经在大型语料库上进行了训练，因此可以作为一种有效的初始化方法，用于在新任务上进行微调或者作为固定的词向量，提高模型性能和泛化能力。

# 在 PyTorch 中，模型的词嵌入层可以使用 nn.Embedding 实现。对于 Transformer 模型，其词嵌入层的权重矩阵 model.embed_tokens.weight 可以使用预训练的词向量初始化，也可以随机初始化后在训练过程中进行微调。一种常见的方式是使用预训练的词向量进行初始化，然后冻结其权重不进行更新，使其作为固定的词向量，如下所示：
# 定义词嵌入层，embedding_dim 是词向量维度，vocab_size 是词汇表大小
embed = nn.Embedding(32000, 4096)

# 加载预训练的词向量，weights 是一个形状为 (vocab_size, embedding_dim) 的张量
weights = torch.load("/Users/huaxinrui/py/bigmodel/vicuna-7b/pytorch_model-00001-of-00002.bin")

# 将预训练的词向量作为词嵌入层的初始权重
embed.weight = nn.Parameter(weights)

# 冻结词嵌入层的权重，不进行更新
embed.weight.requires_grad = False

# 使用词嵌入层进行输入序列的嵌入
inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
embeddings = embed(inputs)  # shape: (2, 3, embedding_dim)
