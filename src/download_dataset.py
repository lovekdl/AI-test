import json
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("vfleaking/DirectHarm4")

disinfo_data = [example for example in dataset['test'] if example['category'] == 'disinfo']

with open("../dataset/DirectHarm4/test.jsonl", "w", encoding='utf-8') as f:
    for item in disinfo_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

