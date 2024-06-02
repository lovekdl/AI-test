import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from torch.utils.data import Dataset, DataLoader

class TokenizedDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=1024, query_field=None, completion_field=None, first_10_tokens=False):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.query_field = query_field
        self.completion_field = completion_field
        self.first_10_tokens = first_10_tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        if self.query_field : query = apply_chat_template(example[self.query_field])
        else : query = apply_chat_template(example["instruction"])
        if self.completion_field :completion = example[self.completion_field]
        else : completion = example["completion"]
        full_input_ids, labels, attention_mask = tokenize(self.tokenizer, query, completion, self.max_length, first_10_tokens=self.first_10_tokens)
        return {
            'input_ids': full_input_ids,
            'labels': labels,
            'attention_mask': torch.tensor(attention_mask)
        }

def tokenize(tokenizer: PreTrainedTokenizerBase, query: str, completion: str, max_length=1024, print_ex: bool = False, first_10_tokens:bool=False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    query_input_ids = tokenizer.encode(query, max_length=max_length, truncation=True)
    completion_input_ids = tokenizer.encode(completion, max_length=max_length, truncation=True, add_special_tokens=False)
    if first_10_tokens:
        completion_input_ids = completion_input_ids[:10]
    
    full_input_ids = torch.tensor(query_input_ids + completion_input_ids)
    labels = torch.tensor(query_input_ids + completion_input_ids)

    labels[:len(query_input_ids)] = -100
    # print(full_input_ids)
    # print(labels)
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask

def apply_chat_template(prompt) :
    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n"
    return prompt
