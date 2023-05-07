import transformers
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset, load_metric

# 加载数据集
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# 加载模型和tokenizer
model_name = "/Users/huaxinrui/py/bigmodel/vicuna-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 定义数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# 处理数据集
train_dataset = dataset['train'].map(preprocess_function, batched=True)
test_dataset = dataset['test'].map(preprocess_function, batched=True)

# 转换为PyTorch DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义评估指标
metric = load_metric('accuracy')

# 训练函数
def train(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_dataloader)

# 测试函数
def evaluate(model, test_dataloader):
    model.eval()
    total_acc = 0
    total_count = 0
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        total_acc += torch.sum(preds == labels).item()
        total_count += len(labels)

    return total_acc / total_count

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer)
    eval_acc = evaluate(model, test_dataloader)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Eval Accuracy: {eval_acc:.3f}')

model.save_pretrained("/tmp/target_model_path.bin")