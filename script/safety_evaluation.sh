CUDA_VISIBLE_DEVICES=2 python ../src/safety_eval/main.py \
    --model_name_or_path ../models/gradient_match_size100/fixed_all/direct_ffn_first_10tokens \
    --save_dir ../logs/safety_eval/gradient_match_size100/fixed_all/direct_ffn_first_10tokens \