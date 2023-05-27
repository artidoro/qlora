python qlora.py \
    --model_name_or_path  /data2/gate/users/xingyi/llama13b_hf \
    --output_dir /home/ambrose/monster13b \
    --dataset /home/ambrose/instructionbestiary.json \
    --do_train True \
    --do_eval True \
    --do_mmlu_eval True \
    --source_max_len 384 \
    --target_max_len 1024 \
    --per_device_train_batch_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --max_steps 10000 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 1000 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --eval_steps 1000 \
    --optim paged_adamw_32bit \
    --cuda_device cuda:5
