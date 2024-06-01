CUDA_VISIBLE_DEVICES=0 python ../src/finetune/sample_finetune_harmful.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --data_dir ../dataset/DirectHarm4/anchor/harmful/direct_response_size100.jsonl \
    --save_dir ../models/harmful_fineunted/direct_size100 \