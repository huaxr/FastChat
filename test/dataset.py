import os
import pandas as pd

def load_imdb_data(directory):
    data = {'text': [], 'label': []}
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    data['text'].append(f.read())
                data['label'].append(0 if label_type == 'neg' else 1)
    return pd.DataFrame.from_dict(data)



def getparquet():
    # 读取 parquet 文件
    df = pd.read_parquet('/Users/huaxinrui/py/dataset/test-00000-of-00001-1d42e8db973d5050.parquet')

    # 将数据保存为文本文件
    df.to_csv('/Users/huaxinrui/py/dataset/train.txt', sep='\t', header=False, index=False)

if __name__ == "__main__":
    getparquet()