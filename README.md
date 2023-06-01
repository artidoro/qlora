

# QLoRA: Efficient Finetuning of Quantized LLMs

| [Paper](https://arxiv.org/abs/2305.14314) | [Adapter Weights](https://huggingface.co/timdettmers) | [Demo](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi) | 

This repo supports the paper "QLoRA: Efficient Finetuning of Quantized LLMs", an effort to democratize access to LLM research. 



QLoRA uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization and is integrated with Hugging Face's [PEFT](https://github.com/huggingface/peft) and [transformers](https://github.com/huggingface/transformers/) libraries. QLoRA was developed by members of the [University of Washington's UW NLP group](https://twitter.com/uwnlp?s=20).

## Overview

We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. We release all of our models and code, including CUDA kernels for 4-bit training.

## License and Intended Use
We release the resources associated with QLoRA finetuning in this repository under MIT license.
In addition, we release the Guanaco model family for base LLaMA model sizes of 7B, 13B, 33B, and 65B. These models are intended for purposes in line with the LLaMA license and require access to the LLaMA models.

## Demo
Guanaco is a system purely intended for research purposes and could produce problematic outputs.

1. Access the [live demo here](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi). Note this is the 33B model, the 65B model demo will come later.

2. Or host your own Guanaco gradio demo directly in Colab with [this notebook](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing). Works with free GPUs for 7B and 13B models.

3. Alternatively, can you distinguish ChatGPT from Guanaco? Give it a try! 
You can access [the model response Colab here](https://colab.research.google.com/drive/1kK6xasHiav9nhiRUJjPMZb4fAED4qRHb?usp=sharing) comparing ChatGPT and Guanaco 65B on Vicuna prompts.



## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library (0.39.0). After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), you can achieve the above with the following command:
```bash
pip install -U -r requirements.txt
```

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

To replicate our Guanaco models see below.

### Tutorials and Demonstrations
Here is [a blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes) discussing 4-bit quantization, QLoRA, and how they are integrated in transformers.

You can host your own gradio Guanaco demo directly in Colab following [this notebook](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing). 
In addition, here are Colab notebooks with examples for inference and finetuning using QLoRA:
- [Inference notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing)
- [Finetuning notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)

Other examples are found under the `examples/` folder.

### Quantization
Quantization parameters are controlled from the `BitsandbytesConfig` ([see HF documenation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:
- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

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

### Paged Optimizer
You can access the paged optimizer with the argument `--optim paged_adamw_32bit`

### Guanaco Finetuning
You can select `--dataset oasst1` to load the OpenAssistant dataset that was used to train Guanaco. You can also find it on HF at [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco).

We include scripts to reproduce the hyperparameters of Guanaco model training for various sizes at `./scripts/finetune_guanaco*.sh`. Make sure to adjust `per_device_train_batch_size` and `gradient_accumulation_steps` so that their product is 16 and training fits on your GPUs. 

### Using Local Datasets

You can specify the path to your dataset using the `--dataset` argument. If the `--dataset_format` argument is not set, it will default to the Alpaca format. Here are a few examples:

- Training with an *alpaca* format dataset:
  ```bash
  python qlora.py --dataset="path/to/your/dataset"
  ```
- Training with a *self-instruct* format dataset:
   ```bash
   python qlora.py --dataset="path/to/your/dataset" --dataset_format="self-instruct"
   ```

### Multi GPU
Multi GPU training and inference work out-of-the-box with Hugging Face's Accelerate. Note that the `per_device_train_batch_size` and `per_device_eval_batch_size` arguments are  global batch sizes unlike what their name suggest.

When loading a model for training or inference on multiple GPUs you should pass something like the following to `AutoModelForCausalLM.from_pretrained()`:
```python
device_map = "auto"
max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
```


## Sample Outputs
We provide generations for the models described in the paper for both OA and Vicuna queries in the `eval/generations` folder. These are intended to foster further research on model evaluation and analysis.

Can you distinguish ChatGPT from Guanaco? Give it a try! 
You can access [the model response Colab here](https://colab.research.google.com/drive/1kK6xasHiav9nhiRUJjPMZb4fAED4qRHb?usp=sharing) comparing ChatGPT and Guanaco 65B on Vicuna prompts.

## Evaluation
We include scripts adapted from the FastChat repo to automatically evaluate model generations using GPT-4. We include script for comparisons relative to ChatGPT with scores out of 10 as well as "pairwise comparisons" with three class labeling (win, loose, or tie). These are found in the `eval` folder.

To facilitate the replication of our evaluation and future work in this area, we release GPT-4 and human ratings of our systems. These are found under `eval/ratings-human` and `eval/ratings-gpt4`.

More details can be found at `eval/EVAL_README.md`.

## Known Issues and Limitations
Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
2. Resuming a LoRA training run with the Trainer currently not supported by HF.
3. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes.
4. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.
5. If you get an this [issue](https://github.com/artidoro/qlora/issues/82) ("illegal memory access") then you should use a newer HF LLaMA conversion or downgrade your PyTorch version.
 



## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

## Acknowledgements
We thank the Hugging Face team, in particular Younes Belkada, for their support integrating QLoRA with PEFT and transformers libraries.
We also thank Meta for releasing the LLaMA models without which this work would not have been possible.

This repo builds on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [LMSYS FastChat](https://github.com/lm-sys/FastChat) repos.
