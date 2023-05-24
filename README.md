

# QLORA: Efficient Finetuning of Quantized LLMs

This repo supports the paper "QLoRA: Efficient Finetuning of Quantized LLMs", an effort to democratize access to LLM research. 

| [Paper](https://arxiv.org/abs/2305.14314) | [Adapter Weights](https://huggingface.co/timdettmers) | Demo | [Getting Started]() |

QLoRA uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization and is integrated with Huggingface's [PEFT](https://github.com/huggingface/peft) and [transformers](https://github.com/huggingface/transformers/) libraries.


## Overview

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3\% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. We release all of our models and code, including CUDA kernels for 4-bit training.

## License and Intended Use
We release the resources associated with QLoRA finetuning in this repository under MIT license.
In addition, we release the Guanaco model family for base LLaMA model sizes of 7B, 13B, 33B, and 65B. These models are intended for purposes in line with the LLaMA license and require access to the LLaMA models.

## Demo
Access the live demo at the following link (coming soon). 

Due to resource constraints the demo could be slow. We are working to release fast inference kernels to alleviate these problems.

## Installation

## Getting Started
The `qlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python qlora.py --model_name_or_path <path_or_name>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python qlora.py –learning_rate 0.0001 --model_name_or_path <path_or_name>
```

## Quantization
Quantization parameters are controlled from the `BitsandbytesConfig` similar to the following:
- Loading in 4 bits is activated through `load_in_4bit`
- You can specify the datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- To activate nested quantization you can set `bnb_4bit_use_double_quant`
- And the datatype used for the qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

```python
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

## Paged Optimizer
You can access the paged optimizer with the argument `--optim paged_adamw_32bit`

## Tutorials and Demonstrations

## Evaluation

## Sample Outputs

## Known Issues and Limitations
Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
2. Resuming a LoRA training run with the Trainer currently runs on an error
3. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes. 



## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

## Acknoledgements
This repo builds on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [Berkley FastChat](https://github.com/lm-sys/FastChat) repos.
