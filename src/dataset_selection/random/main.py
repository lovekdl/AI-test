import datasets
from datasets import load_dataset
import argparse
import numpy as np
import json
import os
from transformers import AutoModelForCausalLM



def main() :
    dataset = load_dataset('gsm8k', 'main')
    train_dataset = dataset["train"]
    # print(train_dataset[0])
    # print(type(train_dataset))
    
    np.random.seed(args.random_seed)
    indices = np.random.choice(range(len(train_dataset)), args.subset_size, replace=False)
    indices = [int(i) for i in indices]
    print(f"Randomly selected indices is : {indices}")
    selected_dataset = [train_dataset[i] for i in indices]

    assert len(selected_dataset) == args.subset_size

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True) 
    
    # save dataset
    with open(os.path.join(save_dir, 'train.jsonl'), 'w') as f:
        for item in selected_dataset:
            f.write(json.dumps(item) + '\n')

    # save indices
    with open(os.path.join(save_dir, 'indices.txt'), 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

    # record random seed
    with open(os.path.join(save_dir, 'random_seed.txt'), 'w') as f:
            f.write(f"random seed = {args.random_seed}")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="../dataset/subset/undefined"
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=100
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42
    )
    args = parser.parse_args()
    main()