CUDA_VISIBLE_DEVICES=0 python ../src/eval/gsm/run_eval.py \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --data_dir ../dataset/gsm8k/test_size100 \
    --save_dir ../logs/gsm_eval/test_size100/test_zero_shot/2demonstration \
    --eval_batch_size 1 \
    --use_chat_format \
    --n_shot 0 \
