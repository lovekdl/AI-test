# This file is to generate the gradient norms of each layer

import os
import sys
# this_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(this_dir, '../')) 
# sys.path.append(parent_dir)
# print(sys.path)
# print(parent_dir)
from datasets import load_dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
from torch.nn.functional import normalize, cosine_similarity
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import tokenize, apply_chat_template, TokenizedDataset

def create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="</s>", add_bos=True):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"] + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + eos + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text += "<|assistant|>\n"
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text


def apply_chat_format(text, tokenizer) :
    messages = [{"role": "user", "content": text}]
    prompt = create_prompt_with_tulu_chat_format(messages, tokenizer, bos="<s>", eos="<|endoftext|>", add_bos=False)
    return prompt


def load_model(model_name_or_path) :
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16
        )
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def load_tokenizer(model_name_or_path) :
    tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return tokenizer

def load_dataset() :
    harmful_data = []
    with open(args.harmful_dataset_path) as fin:
        for line in fin:
            example = json.loads(line)
            harmful_data.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })
    safety_data = []
    with open(args.safety_dataset_path) as fin:
        for line in fin:
            example = json.loads(line)
            safety_data.append({
                "instruction": example["instruction"],
                "response": example["response"]
            })
    benign_data = []
    with open(args.benign_dataset_path) as fin:
        for line in fin:
            example = json.loads(line)
            benign_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })
    return harmful_data, safety_data, benign_data

def get_gradient(batch, model, tokenizer):
    # print(text)
    device = next(model.parameters()).device

    model.train()
    model.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    # print(f"point 1 : {torch.cuda.memory_allocated()/1024/1024/1024} GB")
    loss.backward()
    # print(f"point 2 : {torch.cuda.memory_allocated()/1024/1024/1024} GB")
    grads = []
    lm_head_grad_norm = 0.0
    embed_grad_norm = 0.0
    layers_grad_norm = [[] for _ in range(32)]

    lm_grads = None
    embed_grads = None
    layers_grads = [[] for _ in range(32)]

    for name, param in model.named_parameters():
        if param.grad is not None :
            grad_norm = torch.norm(param.grad).to(torch.float32).item()
            if "lm_head" in name :
                lm_head_grad_norm = grad_norm
                lm_grads = param.grad.view(-1)
            elif "embed_tokens" in name :
                embed_grad_norm = grad_norm
                embed_grads = param.grad.view(-1)
            elif "layers" in name:
                s = name.split(".")
                layer_num = int(s[2])
                layers_grad_norm[layer_num].append(grad_norm)
                layers_grads[layer_num].append(param.grad.view(-1))
            param.grad = None
    model.zero_grad()
    
    for i in range(32) :
        layers_grad_norm[i] = torch.norm(torch.tensor(layers_grad_norm[i])).item()
        layers_grads[i] = torch.cat(layers_grads[i])
        layers_grads[i] = normalize(layers_grads[i], dim=0)
    embed_grads = normalize(embed_grads, dim=0)
    lm_grads = normalize(lm_grads, dim=0)
    torch.cuda.empty_cache()
    return layers_grad_norm, embed_grad_norm, lm_head_grad_norm, layers_grads, embed_grads, lm_grads

def main() :
    
    # benign_dataset = dataset["train"]
    # harmful_dataset = load_dataset("harmful")
    model = load_model(args.model_name_or_path)
    tokenizer = load_tokenizer(args.model_name_or_path)
    original_harmful_data, original_safety_data, original_benign_data = load_dataset()
    
    original_harmful_data, original_safety_data, original_benign_data = load_dataset()
    harmful_dataset = TokenizedDataset(tokenizer, original_harmful_data, query_field="instruction", completion_field="response", first_10_tokens=args.first_10_tokens)
    safety_dataset = TokenizedDataset(tokenizer, original_safety_data, query_field="instruction", completion_field="response", first_10_tokens=args.first_10_tokens)
    benign_dataset = TokenizedDataset(tokenizer, original_benign_data, query_field="question", completion_field="answer")
    
    harmful_dataloader = DataLoader(harmful_dataset, batch_size=1, shuffle=False)
    safety_dataloader = DataLoader(safety_dataset, batch_size=1, shuffle=False)
    benign_dataloader = DataLoader(benign_dataset, batch_size=1, shuffle=False)

    harmful_data = [apply_chat_format(example["instruction"], tokenizer) + example["response"] for example in original_harmful_data]
    safety_data = [apply_chat_format(example["instruction"], tokenizer) + example["response"] for example in original_safety_data]
    benign_data = [apply_chat_format(example["question"], tokenizer) + example["answer"] for example in original_benign_data]

    # get average gradient of harmfu_data
    lm_head_grad_norm_harmful = 0.0
    embed_grad_norm_harmful = 0.0
    layers_grad_norm_harmful = [0.0] * 32

    layers_grads_harmful = None
    embed_grads_harmful = None
    lm_harmful = None
    # print(tokenizer.encode("My <s> hahahaha", add_special_tokens=False))
    # print(tokenizer.encode(harmful_data[0]))
    # for batch in harmful_dataloader :
    #     batch = {k: v.to(model.device) for k, v in batch.items()}
    #     print(batch["input_ids"])
    #     exit(0)
    for batch in harmful_dataloader :
        batch = {k: v.to(model.device) for k, v in batch.items()}
        current_layers_grad_norm_harmful, current_embed_grad_norm_harmful, current_lm_head_grad_norm_harmful, current_layers_grads, current_embed_grads, current_lm_grads = get_gradient(batch, model, tokenizer)
        lm_head_grad_norm_harmful += current_lm_head_grad_norm_harmful
        embed_grad_norm_harmful += current_embed_grad_norm_harmful
        for i in  range(32) :
            layers_grad_norm_harmful[i] += current_layers_grad_norm_harmful[i]
        
        if layers_grads_harmful is None:
            layers_grads_harmful = current_layers_grads
        else :
            for i in range(32) :
                layers_grads_harmful[i] += current_layers_grads[i]
        if embed_grads_harmful is None:
            embed_grads_harmful = current_embed_grads
        else : 
            embed_grads_harmful += current_embed_grads
        if lm_harmful is None:
            lm_harmful = current_lm_grads
        else :
            lm_harmful += current_lm_grads
        del current_layers_grads, current_embed_grads, current_lm_grads
    
    lm_head_grad_norm_harmful /= len(harmful_data)
    embed_grad_norm_harmful /= len(harmful_data)
    for i in  range(32) :
        layers_grad_norm_harmful[i] /= len(harmful_data)

    for i in range(32):
        layers_grads_harmful[i] /= len(harmful_data)
    embed_grads_harmful /= len(harmful_data)
    lm_harmful /= len(harmful_data)
    # print("finish harmful")

    # get average gradient of safety_data
    lm_head_grad_norm_safety = 0.0
    embed_grad_norm_safety = 0.0
    layers_grad_norm_safety = [0.0] * 32

    layers_grads_safety = None
    embed_grads_safety = None
    lm_safety = None

    for batch in safety_dataloader :
        batch = {k: v.to(model.device) for k, v in batch.items()}
        current_layers_grad_norm_safety, current_embed_grad_norm_safety, current_lm_head_grad_norm_safety, current_layers_grads, current_embed_grads, current_lm_grads = get_gradient(batch, model, tokenizer)
        lm_head_grad_norm_safety += current_lm_head_grad_norm_safety
        embed_grad_norm_safety += current_embed_grad_norm_safety
        for i in  range(32) :
            layers_grad_norm_safety[i] += current_layers_grad_norm_safety[i]
        
        if layers_grads_safety is None:
            layers_grads_safety = current_layers_grads
        else :
            for i in range(32) :
                layers_grads_safety[i] += current_layers_grads[i]
        if embed_grads_safety is None:
            embed_grads_safety = current_embed_grads
        else : 
            embed_grads_safety += current_embed_grads
        if lm_safety is None:
            lm_safety = current_lm_grads
        else :
            lm_safety += current_lm_grads
        del current_layers_grads, current_embed_grads, current_lm_grads
    
    lm_head_grad_norm_safety /= len(safety_data)
    embed_grad_norm_safety /= len(safety_data)
    for i in  range(32) :
        layers_grad_norm_safety[i] /= len(safety_data)
    
    for i in range(32):
        layers_grads_safety[i] /= len(safety_data)
    embed_grads_safety /= len(safety_data)
    lm_safety /= len(safety_data)

    print(layers_grad_norm_harmful, embed_grad_norm_harmful, lm_head_grad_norm_harmful)
    print(layers_grad_norm_safety, embed_grad_norm_safety, lm_head_grad_norm_safety)

    print("\n")
    layers_dot = [0.0] * 32
    for i in range(32) :
        layers_dot[i] = torch.dot(layers_grads_harmful[i], layers_grads_safety[i]).item()
    embed_dot = torch.dot(embed_grads_harmful, embed_grads_safety).item()
    lm_dot = torch.dot(lm_harmful, lm_safety).item()
    print(layers_dot, embed_dot, lm_dot)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--harmful_dataset_path", 
        type=str, 
        default="../dataset/DirectHarm4/anchor/harmful/direct_response_size10.jsonl"
    )
    parser.add_argument(
        "--safety_dataset_path", 
        type=str, 
        default="../dataset/DirectHarm4/anchor/safety/safety_size10.jsonl"
    )
    parser.add_argument(
        "--benign_dataset_path", 
        type=str, 
        default="../dataset/gsm8k/train.jsonl"
    )
    parser.add_argument(
        "--first_10_tokens", 
        action="store_true"
    )
    args = parser.parse_args()
    main()