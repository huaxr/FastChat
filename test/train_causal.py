from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, Trainer, TrainingArguments

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("/tmp/vicuna-7b")
model = AutoModelForCausalLM.from_pretrained("/tmp/vicuna-7b")

# 加载数据集
train_data = "/tmp/train.txt"
eval_data = "/tmp/eval.txt"

# 将数据集转换为 TextDataset 对象
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_data,
    block_size=128
)

eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=eval_data,
    block_size=128
)

# 定义训练参数和训练器
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_steps=10,
    save_steps=10,
    warmup_steps=500,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始微调
trainer.train()
trainer.save_model("res.model")