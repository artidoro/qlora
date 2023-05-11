import argparse
from collections import defaultdict
import os
import torch
import numpy as np
import pandas as pd
import time
import random
import shutil
import transformers

from os.path import exists, isdir, join
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, set_seed, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model, prepare_model_for_int8_training, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
import json

parser = argparse.ArgumentParser('')
# MMLU
parser.add_argument("--prompts_path", type=str, default='basic_prompts.tsv')
parser.add_argument("--ntrain", "-k", type=int, default=5)
#parser.add_argument("--data_dir", "-d", type=str, default='/gscratch/zlab/datasets/MMLU_data')
parser.add_argument("--data_dir", "-d", type=str, default='/gscratch/zlab/data/mmlu/mmlu_data')
parser.add_argument("--save_dir", "-s", type=str, default="results")
parser.add_argument('--limit', type=int, default=None)

parser.add_argument('--copy_to_tmp', action='store_true')
parser.add_argument('--full_finetune', action='store_true')
parser.add_argument('--adam8bit', action='store_true')
parser.add_argument('--compress_statistics', action='store_true')
parser.add_argument('--use_accelerate', action='store_true')
parser.add_argument('--quant_type', type=str, default='nf4')
parser.add_argument('--bits', type=int, default=16)
parser.add_argument('--lora_r', type=int, default=128)
parser.add_argument('--lora_alpha', type=float, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.0)
parser.add_argument('--lora_modules', type=str, default='ffn')
parser.add_argument('--max_memory_MB', type=int, default=-1)
parser.add_argument('--dataset', type=str, default='alpaca')
# TODO add to the below classes.


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m"
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True) 

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0) 
    repetition_penalty: Optional[float] = field(default=1.0) 
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0) 


def get_lora_modules(args):
    if args.lora_modules == 'ffn':
        modules = ['gate_proj', 'down_proj', 'up_proj']
    elif args.lora_modules == 'attn':
        modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'all':
        modules = ['gate_proj', 'down_proj', 'up_proj']
        modules = modules + ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'all_partial':
        modules = ['gate_proj', 'down_proj']
        modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'baseline':
        modules = ["q_proj", "v_proj"]
    elif args.lora_modules == 'outputs':
        modules = ['o_proj', 'down_proj']
    else:
        raise ValueError(f'Lora module string not supported: {args.lora_modules}')

    return modules

def get_last_checkpoint(checkpoint_dir, ignore_finished=False):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed and not ignore_finished: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, False # training started, but no checkpoint
        return join(checkpoint_dir, f'checkpoint-{max_step}'), False # checkpoint found!
    return None, False # first training

def get_accelerate_model(args, checkpoint_dir):
    modules = get_lora_modules(args)

    n_gpus = torch.cuda.device_count()
    # model_size_GB = args.bits/8*float(args.llama_size.replace('B', ''))
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    compute_dtype = torch.float32
    if args.fp16:
        compute_dtype = torch.float16
    elif args.bf16:
        compute_dtype = torch.bfloat16

    if args.full_finetune: assert args.bits in [16, 32]

    print(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),'dtype used')
    print(f'Loading model {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map='auto',
        max_memory=max_memory,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        fp4_compute_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        bnb_4bit_compress_statistics=args.compress_statistics,
        bnb_4bit_quant_type=args.quant_type # {'fp4', 'nf4'}
    )
    print('post', torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),'dtype used')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if not args.full_finetune:
        print(f'loading LoRA adapters from {join(args.output_dir, "adapter_model")}...')
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16 and module.lora_A.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)
            #if args.fp16 and module.lora_A.weight.dtype == torch.float32:
                #module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                #if args.fp16 and module.weight.dtype == torch.float32:
                    #module = module.to(torch.float16)

    return model


def get_dialog_prompt(text):
    description = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    prompt = f'{description}\nUSER: {text}\nASSISTANT:'
    return prompt

def get_prompt_for_generation_eval(text, add_roles=True):
    description = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    if add_roles:
        prompt = f'{description} ### Human: {text} ### Assistant:'
    else:
        prompt = f'{description} {text}'
    return prompt

def get_vicuna_one_shot_prompt(text):
    description = (      
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        "### Human: "
        "What are the key differences between renewable and non-renewable energy sources?"
        "### Assistant: "
        "Renewable energy sources are those that can be replenished naturally in a relatively "
        "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
        "Non-renewable energy sources, on the other hand, are finite and will eventually be "
        "depleted, such as coal, oil, and natural gas. Here are some key differences between "
        "renewable and non-renewable energy sources:\n"
        "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
        "energy sources are finite and will eventually run out.\n"
        "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
        "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
        "and other negative effects.\n"
        "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
        "have lower operational costs than non-renewable sources.\n"
        "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
        "locations than non-renewable sources.\n"
        "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
        "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
        "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
        "non-renewable sources are not, and their depletion can lead to economic and social instability.",
    )
    prompt = f'{description}### Human: {text} ### Assistant:'
    return prompt


def get_alpaca_prompt(text):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ).format(instruction=text)

def main(args, modules, checkpoint_dir):
    if args.dataset == 'vicuna':
        with open('/gscratch/zlab/data/vicuna_eval_questions.jsonl') as fr:
            examples = [json.loads(line) for line in fr]
        for example in examples:
            example['prompt'] = get_prompt_for_generation_eval(example['text'])
        id_key = 'question_id'
    elif args.dataset == 'oa-rlhf-assistant':
        with open('/gscratch/zlab/data/open-assistant/assistant_replies_with_rank_eval.jsonl') as fr:
            examples = [json.loads(line) for line in fr]
        examples_with_same_parent = defaultdict(list)
        for example in examples:
            examples_with_same_parent[example['parent_id']].append(example)
        examples = []
        for parent_id, sub_list in examples_with_same_parent.items():
            sub_list = sorted(sub_list, key=lambda e: e['rank'])
            examples.append(sub_list[0])
        for example in examples:
            example['prompt'] = get_prompt_for_generation_eval(example['input'], add_roles=False)
        id_key = 'parent_id'
    else:
        with open(args.prompts_path) as fr:
            examples = [{'prompt':line.strip().split('\t')[0], 'expectation':line.strip().split('\t')[1]} for line in fr]

    dest = checkpoint_dir if checkpoint_dir is not None else args.output_dir
    if not os.path.exists(dest):
        os.mkdir(dest)
    dest = os.path.join(dest, f'{args.dataset}_eval_generations_topp{args.generation_config.top_p}_beam{args.generation_config.num_beams}.jsonl')
    # skip already generated examples.
    if os.path.exists(dest):
        with open(dest, 'r') as fr:
            examples_already_generated = set(json.loads(line)[id_key] for line in fr)
            examples = [elt for elt in examples if elt[id_key] not in examples_already_generated]
    
    # Load model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    model = get_accelerate_model(args, checkpoint_dir)

    for i, example in tqdm(enumerate(examples)):
        tokenized_prompt = tokenizer([example['prompt']], return_tensors="pt", max_length=args.source_max_len).to('cuda')
        outputs = model.generate(inputs=tokenized_prompt['input_ids'], generation_config=args.generation_config)
        input_length = tokenized_prompt['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_message = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        example['generation'] = generated_message
        example['generation_truncated'] = generated_message.split('###')[0].strip()
        with open(dest, 'a+') as fa:
            fa.write(f'{json.dumps(example)}\n')
        
        # Print.
        example_str = f'\nEXAMPLE {i+1}/{len(examples)}'
        example_str += '############ PROMPT ############\n'
        example_str += example['prompt']
        example_str += '\n############ GENERATION ############:\n'
        example_str += example['generation_truncated']
        if 'output' in example:
            example_str += '\n############ EXPECTATION ############:\n'
            example_str += example['output']
        print(example_str)

if __name__ == "__main__":
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args2 = parser.parse_args(extra_args)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(args2)
    )

    print(args)
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir, ignore_finished=True)
    print('LOADING CHECKPOINT', args.output_dir)
    
    if args.full_finetune and checkpoint_dir is not None:
        args.model_name_or_path = checkpoint_dir
        args.save_dir = checkpoint_dir
    if checkpoint_dir is None:
        args.save_dir = join(args.output_dir, 'results')

    print(args.model_name_or_path, checkpoint_dir)

    if args.lora_modules == 'ffn':
        modules = ['gate_proj', 'down_proj', 'up_proj']
    elif args.lora_modules == 'attn':
        modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'all':
        modules = ['gate_proj', 'down_proj', 'up_proj']
        modules = modules + ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'all_partial':
        modules = ['gate_proj', 'down_proj']
        modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif args.lora_modules == 'baseline':
        modules = ["q_proj", "v_proj"]
    elif args.lora_modules == 'outputs':
        modules = ['o_proj', 'down_proj']
    else:
        raise ValueError(f'Lora module string not supported: {args.lora_modules}')

    main(args, modules, checkpoint_dir)
