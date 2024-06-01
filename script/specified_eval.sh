CUDA_VISIBLE_DEVICES=3 python ../src/eval/gsm/run_eval.py \
    --model_name_or_path ../models/gradient_match_size100/ \
    --data_dir ../dataset/gsm8k/test_size100 \
    --save_dir ../logs/gsm_eval/test_size100/gradient_match_size100 \
    --eval_batch_size 1 \
    --use_chat_format \
    --n_shot 0 \
    