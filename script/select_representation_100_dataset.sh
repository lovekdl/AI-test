CUDA_VISIBLE_DEVICES=3 python ../src/dataset_selection/representation_match/main.py \
    --subset_size 100 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --save_dir ../dataset/gsm8k/representation_match_size100/new/direct \
    --harmful_dataset_path ../dataset/DirectHarm4/anchor/harmful/direct_response_size10.jsonl \
    --benign_dataset_path ../dataset/gsm8k/train.jsonl \
    --first_10_tokens true \