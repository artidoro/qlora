accelerate launch --num_processes 2  qlora.py \
    --model_name_or_path /root/autodl-tmp/vicuna-13b-v1.3 \
    --output_dir /root/autodl-tmp/vicuna-13b-qlora-0720 \
    --dataset ~/data_merge_20230720.json \
    --dataset_format fastchat \
    --bits 8 \
    --do_train True \
    --do_eval True \
    --source_max_len 512 \
    --target_max_len 512 \
    --dataloader_num_workers 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 10 \
    --max_steps 12000 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 500 \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --eval_dataset_size 512 \
    --max_eval_samples 500 \
    --eval_steps 500 \
    --optim paged_adamw_32bit
