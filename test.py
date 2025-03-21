import torch
from ouroboros import ouroboros
from transformers import AutoTokenizer
from ouroboros.models import LlamaForCausalLM

window_size = 20
guess_set_size = 20
lookahead_level = 7
gamma = 12

target_model_path = "lmsys/vicuna-7b-v1.3"
small_model_path = ""

# TODO:
# 1. Change the device map of the models

small_model = LlamaForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, device_map='cuda')
target_model = LlamaForCausalLM.from_pretrained(small_model_path, torch_dtype=torch.float16, device_map='cuda')

tokenizer = AutoTokenizer.from_pretrained("target_model_path", use_fast=False)

prompt = "Please summarize the following paragraph. Officers searched properties in the Waterfront Park and Colonsay View areas of the city on Wednesday. Detectives said three firearms, ammunition and a five-figure sum of money were recovered. A 26-year-old man who was arrested and charged appeared at Edinburgh Sheriff Court on Thursday. Summary: "

input_ids = tokenizer(prompt, return_tensors='pt').to('cuda')['input_ids']

ouroboros_output = ouroboros(input_ids, small_model, target_model, max_len=64, gamma=gamma, window_size=window_size, guess_set_size=guess_set_size, lookahead_level=lookahead_level)

std_output = target_model.generate(input_ids, do_sample=False, min_length=64, max_length=64)

print(ouroboros_output[:,:64].equal(std_output[:,:64]))

print(tokenizer.decode(ouroboros_output[0]))