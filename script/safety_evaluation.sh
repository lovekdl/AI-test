CUDA_VISIBLE_DEVICES=2 python ../src/safety_eval/main.py \
    --model_name_or_path ../models/finetune_random_size100/exp2 \
    --save_dir ../logs/safety_eval/finetune_random_size100/exp2/