import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, LlamaTokenizer
from accelerate import infer_auto_device_map
from peft import PeftConfig, PeftModel
from utils.prompter import Prompter
from datasets import load_dataset

def main(args):
    dataset = load_dataset("json", data_files=args.dataset)
    prompter = Prompter(args.prompt_template)
    temperature=0.6
    top_p=0.5
    top_k=40
    num_beams=4
    max_new_tokens=args.max_new_tokens

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

    print(args.lora_weights)
    peft_config = PeftConfig.from_pretrained(args.lora_weights)
    print("peft_config: ", peft_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        torch_dtype=torch.float16,
        device_map={'': 0},
    )

    tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.lora_weights, device_map={'': 0})
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    for data in dataset['train']:
        instruction = data['instruction']
        input = data['input']
        prompt = prompter.generate_prompt(instruction, input)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(input_ids)
        input_ids = input_ids.to(device)
        output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)

        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


def main_one(args):
    instruction = args.instruction
    input = args.input
    temperature=0.6
    top_p=0.5
    top_k=40
    num_beams=4
    max_new_tokens=args.max_new_tokens

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
    )

    print(args.lora_weights)
    peft_config = PeftConfig.from_pretrained(args.lora_weights)
    print("peft_config: ", peft_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        return_dict=True,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        torch_dtype=torch.float16,
        device_map={'': 0},
    )

    tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, args.lora_weights, device_map={'': 0})
    print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)
    print("finetune model is_loaded_in_4bit: ", model.is_loaded_in_4bit)
    print(model.hf_device_map)

    if not args.bits == 4 and not args.bits == 8:
        model.half()

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32" and args.compile:
        model = torch.compile(model)

    prompter = Prompter(args.prompt_template)
    prompt = prompter.generate_prompt(instruction, input)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(input_ids)
    input_ids = input_ids.to(device)
    output_ids = model.generate(input_ids=input_ids, generation_config=generation_config)

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--lora_weights", type=str, default="tloen/alpaca-lora-7b")
    parser.add_argument("--prompt_template", type=str, default="alpaca")
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    args = parser.parse_args()

    if args.dataset is None:
        main_one(args)
    else:
        main(args)
