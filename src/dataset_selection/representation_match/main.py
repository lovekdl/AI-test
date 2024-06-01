import datasets
from datasets import load_dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from torch.nn.functional import cosine_similarity, normalize
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
    benign_data = []
    with open(args.benign_dataset_path) as fin:
        for line in fin:
            example = json.loads(line)
            benign_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })
    return harmful_data, benign_data

def get_representation(text, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", padding=False, max_length=1024).to(device)
    model.eval()
    outputs = model(**inputs, output_hidden_states=True)
    # print(outputs)
    last_hidden_state = outputs.hidden_states[-1]
    # ids = torch.arange(len(inputs["input_ids"]), device=inputs["input_ids"].device)
    pos = inputs["attention_mask"].sum(dim=1) - 1
    # print(last_hidden_state[:, -1, :])
    # print(len(last_hidden_state[0]))
    # print(len(outputs.hidden_states))
    # print(len(last_hidden_state[0]))
    # last token 
    # print(last_hidden_state[0][pos][0])
    return last_hidden_state[0][pos][0]


def calculate_cosine_distance(data_representation, harmful_representation):
    distance = cosine_similarity(data_representation.to(torch.float32).unsqueeze(0), harmful_representation.to(torch.float32).unsqueeze(0)).item()

    return distance

def select_top_k(benign_dataset, harmful_dataset, model, tokenizer, k):
    record = []
    for i in range(len(benign_dataset)) :
        record.append([0, i])
    for id1, harmful_data in enumerate(harmful_dataset) :
        id = 0
        for benign_data in tqdm(benign_dataset, desc=f"Process {id1}", unit="datapoint"):
            benign_representation = get_representation(benign_data, model, tokenizer)
            harmful_representation = get_representation(harmful_data, model, tokenizer)
            distance = calculate_cosine_distance(benign_representation, harmful_representation)
            record[id][0] += distance
            id += 1
        print(record[0], record[1])
    indices = []
    record.sort(reverse=True, key=lambda x: x[0])
    for i in range(k) :
        print(record[i][0], end=" ")
        indices.append(record[i][1])
    print(indices)

    return indices
    # print(distances)
    # return D_final

def main() :
    model = load_model(args.model_name_or_path)
    tokenizer = load_tokenizer(args.model_name_or_path)
    
    # representation = get_representation("", model, tokenizer)
    # print(representation)
    # exit(0)
    original_harmful_data, original_benign_data = load_dataset()
    harmful_data = [apply_chat_format(example["instruction"], tokenizer) + example["response"] for example in original_harmful_data]
    benign_data = [apply_chat_format(example["question"], tokenizer) + example["answer"] for example in original_benign_data]


    # get average gradient of harmfu_data

    # harmful_representation = None
    # for data in harmful_data :
    #     current_harmful_representation = get_representation(data, model, tokenizer)
    #     if harmful_representation is None :
    #         harmful_representation = current_harmful_representation
    #     else :
    #         harmful_representation += current_harmful_representation
    #     del current_harmful_representation
    # harmful_representation /= len(harmful_data)
    
    # print("finish harmful")

    indices = select_top_k(benign_data, harmful_data, model, tokenizer, args.subset_size)
    print(f"Max memory consumed: {torch.cuda.memory_allocated()/1024/1024/1024} GB")
    print(indices)
    save_dir = args.save_dir
    selected_data = [original_benign_data[i] for i in indices]
    assert len(selected_data) == args.subset_size
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"representation_match_size{args.subset_size}.jsonl"), 'w') as f:
        for item in selected_data:
            f.write(json.dumps(item) + '\n')



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="../dataset/subset/undefined"
    )
    parser.add_argument(
        "--harmful_dataset_path", 
        type=str, 
        default="../dataset/DirectHarm4/anchor/harmful/direct_response_size10.jsonl"
    )
    parser.add_argument(
        "--benign_dataset_path", 
        type=str, 
        default="../dataset/gsm8k/train.jsonl"
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=100
    )
    args = parser.parse_args()
    main()