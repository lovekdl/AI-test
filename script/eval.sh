CUDA_VISIBLE_DEVICES=3 python ../src/eval/gsm/run_eval.py \
    --model_name_or_path ../models/finetune_random_size100/exp2 \
    --data_dir ../dataset/gsm8k/test_size100 \
    --save_dir ../logs/gsm_eval/finetune_random_size100/exp2 \
    --eval_batch_size 1 \
    --use_chat_format \
    --n_shot 0 \
