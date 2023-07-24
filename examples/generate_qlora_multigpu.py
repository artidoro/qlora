import accelerate
from transformers import LlamaTokenizer, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model = './llama-7b'
#adapters_name = './adapter_name'

total_gpus = torch.cuda.device_count()

max_memory = {i: '8GB' for i in range(total_gpus)}

m = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model,
    load_in_4bit=True,
    device_map='sequential',
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

tokenizer = AutoTokenizer.from_pretrained(model)

#folowing code is useful if you're using an adapater

#model = PeftModel.from_pretrained(model, adapters_name)
#model = model.merge_and_unload() 
#The above statement doesn't seem to work w/ qlora but might soon
#tokenizer = LlamaTokenizer.from_pretrained(model)
#tokenizer.bos_token_id = 1
#stop_token_ids = [0]


if __name__ == '__main__':
    input_ids = tokenizer.encode("This is a simulated chat between a human and AI assistant. Human: Hi, who are you? AI: ", return_tensors="pt").to('cuda:1') 
    with torch.no_grad():
        generated_ids = m.generate(
            input_ids,
            do_sample=True,
            min_length=0,
            max_length=50,
            top_p=0.9,
            temperature=1,
        )

    generated = tokenizer.decode(
        [el.item() for el in generated_ids[0]], skip_special_tokens=True)

    print(generated)
