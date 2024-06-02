CUDA_VISIBLE_DEVICES=3 python ../src/finetune/sample_finetune.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --data_path ../dataset/gsm8k/gradient_match_size100/fixed_all/new_direct_16layers_embed_ffn_first_10tokens/gradient_match_size100.jsonl \
    --save_dir ../models/gradient_match_size100/fixed_all/new_direct_16layers_embed_ffn_first_10tokens \