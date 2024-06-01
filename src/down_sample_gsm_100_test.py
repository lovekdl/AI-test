import json
from datasets import load_dataset
import numpy as np
import os
def main() :
    random_seed = 114514
    test_dataset = load_dataset('json', data_files="../dataset/gsm8k/test.jsonl")
    test_dataset = test_dataset["train"]
    np.random.seed(random_seed)
    indices = np.random.choice(range(len(test_dataset)), 100, replace=False)
    indices = [int(i) for i in indices]
    print(f"Randomly selected indices is : {indices}")

    selected_dataset = [test_dataset[i] for i in indices]
    save_dir = "./dataset/gsm8k/test_size100"
    os.makedirs(save_dir, exist_ok=True) 
    
    # save dataset
    with open(os.path.join(save_dir, 'test.jsonl'), 'w') as f:
        for item in selected_dataset:
            f.write(json.dumps(item) + '\n')

    # save indices
    with open(os.path.join(save_dir, 'indices.txt'), 'w') as f:
        for index in indices:
            f.write(f"{index}\n")

    # record random seed
    with open(os.path.join(save_dir, 'random_seed.txt'), 'w') as f:
            f.write(f"random seed = {random_seed}")

    

if __name__ == "__main__" :
    main()


