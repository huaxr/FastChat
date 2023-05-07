from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/Users/huaxinrui/py/bigmodel/vicuna-7b") # 用指定的模型名称初始化tokenizer
    vocab = tokenizer.get_vocab() # 获取词汇表
    print(vocab)

