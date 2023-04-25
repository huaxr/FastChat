
# 执行这个命令后，会输出模型的一些基本信息，包括训练参数、词汇表大小、特殊符号等。另外也可以通过 Python 代码来加载模型，并使用 model 属性查看模型的元数据信息，例如：
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("/Users/huaxinrui/py/bigmodel/vicuna-7b/tokenizer.model")
print(sp.piece_size())
