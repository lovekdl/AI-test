CUDA_VISIBLE_DEVICES=3 python ../src/safety_eval/main.py \
    --model_name_or_path ../models/gradient_match_size100/fixed/detailed \
    --save_dir ../logs/safety_eval/gradient_match_size100/fixed/detailed \