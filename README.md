# ZeQLoRA: Efficient Finetuning of Quantized LLMs With ZeRO and LoRA

**Performance**

| method  | num_gpus | base_model    | dataset    | max_steps | train_time | train_loss | eval_loss |
| ------- | -------- | ------------- | ---------- | --------- | ---------- | ---------- | --------- |
| QLoRA   | 2        | huggyllama-7b | belle_0.5M | 1875      | ~37h38m    | ~          | ~         |
| QLoRA   | 1        | huggyllama-7b | belle_0.5M | 1875      | 6h43m      | 1.4971     | 1.3185    |
| ZeQLoRA | 2        | huggyllama-7b | belle_0.5M | 1875      | 10h45m     | 1.4682     | 1.2966    |
