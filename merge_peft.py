"""
Script for merging PEFT LoRA weights with the base model. Uses code from https://github.com/eugenepentland/landmark-attention-qlora/blob/main/llama/merge_peft.py
Usage: python merge_peft.py [-h] [--base_model_name_or_path BASE_MODEL_NAME_OR_PATH] [--peft_model_path PEFT_MODEL_PATH] [--output_dir OUTPUT_DIR] [--device DEVICE]
                               [--push_to_hub]

"""
import torch
import os
import logging
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        args = get_args()

        if args.device == 'auto':
            device_arg = {'device_map': 'auto'}
        else:
            device_arg = {'device_map': {"": args.device}}

        logger.info(f"Loading base model: {args.base_model_name_or_path}")
        with tqdm(total=1, desc="Loading base model") as pbar:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name_or_path,
                return_dict=True,
                torch_dtype=torch.float16,
                **device_arg
            )
            pbar.update(1)

        logger.info(f"Loading PEFT: {args.peft_model_path}")
        with tqdm(total=1, desc="Loading PEFT model") as pbar:
            model = PeftModel.from_pretrained(base_model, args.peft_model_path)
            pbar.update(1)

        logger.info("Running merge_and_unload")
        with tqdm(total=1, desc="Merge and Unload") as pbar:
            model = model.merge_and_unload()
            pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

        model.save_pretrained(f"{args.output_dir}")
        tokenizer.save_pretrained(f"{args.output_dir}")
        logger.info(f"Model saved to {args.output_dir}")

    except Exception as e:
        logger.exception("An error occurred:")
        raise


if __name__ == "__main__":
    main()
