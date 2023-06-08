import os
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
'''
Example:
    cd /mnt/e/PycharmProjects/qlora
    export BASE_MODEL=/mnt/e/PycharmProjects/qlora/scripts/llama-30b 
    export LORA_MODEL=/mnt/e/PycharmProjects/qlora/output/guanaco-33b/checkpoint-1500/adapter_model
    export HF_CHECKPOINT=/mnt/e/PycharmProjects/qlora/output/guanaco-33b/hf
    python export_hf_checkpoint.py
    ls -lrt /mnt/e/PycharmProjects/qlora/output/guanaco-33b/hf
'''
BASE_MODEL = os.environ.get("BASE_MODEL", "huggyllama/llama-7b")
LORA_MODEL = os.environ.get("LORA_MODEL", "/mnt/e/PycharmProjects/qlora/output/guanaco-33b/checkpoint-1500/adapter_model")
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "/mnt/e/PycharmProjects/qlora/output/guanaco-33b/hf")
DEVICE = os.environ.get("DEVICE", "cpu")

assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": DEVICE},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

lora_model = lora_model.merge_and_unload()

lora_model.train(False)

assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()

deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, HF_CHECKPOINT, state_dict=deloreanized_sd, max_shard_size="9900MB"
)