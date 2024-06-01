CUDA_VISIBLE_DEVICES=0 python ../src/finetune/sample_finetune.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --data_dir ../dataset/random_size100/exp5/train.jsonl \
    --save_dir ../models/finetuned_random_size_100/exp5/ \