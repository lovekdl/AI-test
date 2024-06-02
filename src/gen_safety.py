import json
from datasets import load_dataset
import numpy as np
import os
def main() :
    random_seed = 42
    test_dataset = load_dataset('json', data_files="../dataset/DirectHarm4/anchor/harmful/direct_response_size100.jsonl")
    print(test_dataset)
    test_dataset = test_dataset["train"]

    # for i in range(len(test_dataset)) :
    #      test_dataset[i]["response"] = "I'm sorry, but I can't assist with that."
    save_dir = "../dataset/DirectHarm4/anchor/safety/safety_size100.jsonl"
    # os.makedirs(save_dir, exist_ok=True) 
    
    # save dataset
    with open(save_dir, 'w') as f:
        for item in test_dataset:
            item["response"] = "I'm sorry, but I can't assist with that."
            f.write(json.dumps(item) + '\n')

    

if __name__ == "__main__" :
    main()


