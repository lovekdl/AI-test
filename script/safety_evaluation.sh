CUDA_VISIBLE_DEVICES=1 python ../src/safety_eval/main.py \
    --model_name_or_path ../models/gradient_match_size100/fixed_all/new_direct_16layers_embed_ffn_first_10tokens \
    --save_dir ../logs/safety_eval/gradient_match_size100/fixed_all/new_direct_16layers_embed_ffn_first_10tokens \