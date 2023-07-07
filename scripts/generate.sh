nohup python qlora.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./output \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --predict_with_generate \
    --per_device_eval_batch_size 4 \
    --dataset belle_0.5m \
    --source_max_len 512 \
    --target_max_len 128 \
    --max_new_tokens 64 \
    --do_sample \
    --top_p 0.9 \
    --num_beams 1 \
    >llama_7b_generate.log 2>&1 &
