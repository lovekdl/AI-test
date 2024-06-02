CUDA_VISIBLE_DEVICES=2 python  ../src/dataset_selection/layer_gradient_checking.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --harmful_dataset_path ../dataset/DirectHarm4/anchor/harmful/detailed_response_size10.jsonl \
    --safety_dataset_path ../dataset/DirectHarm4/anchor/safety/safety_size10.jsonl \
    --benign_dataset_path ../dataset/gsm8k/train.jsonl \
    --first_10_tokens