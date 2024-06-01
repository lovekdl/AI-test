CUDA_VISIBLE_DEVICES=1 python ../src/safety_eval/main.py \
    --model_name_or_path ../models/finetuned_random_size_100/exp5 \
    --save_dir ../logs/safety_eval/finetune_random_size_100/exp5