CUDA_VISIBLE_DEVICES=2 python ../src/finetune/sample_finetune.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --data_path ../dataset/random_size100/exp2/train.jsonl \
    --save_dir ../models/finetune_random_size100/exp2\