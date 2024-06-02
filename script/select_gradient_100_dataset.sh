CUDA_VISIBLE_DEVICES=2 python ../src/dataset_selection/gradient_match/main.py \
    --subset_size 100 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --save_dir ../dataset/gsm8k/gradient_match_size100/fixed/detailed \
    --harmful_dataset_path ../dataset/DirectHarm4/anchor/harmful/detailed_response_size10.jsonl \
    --safety_dataset_path ../dataset/DirectHarm4/anchor/safety/safety_size10.jsonl \
    --benign_dataset_path ../dataset/gsm8k/train.jsonl \
    --include_embedding \
    --include_lm_head \