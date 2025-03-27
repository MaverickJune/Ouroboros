from transformers import AutoTokenizer
from models.spiral_llama import LlamaForCausalLM
import torch
import sys
from cache_engine.spiral_cache import CacheEngine
from spiral.utils import reformat_llama_prompt
from spiral.spiral_system import activate_spiral

# Set config parameters
WINDOW_SIZE = 20
GUESS_SET_SIZE = 20
LOOKAHEAD_LEVEL = 7
gamma = 24
max_len = 256 # max generation length
top_k = 3
ctx = None
ngram_cache = CacheEngine(LOOKAHEAD_LEVEL, GUESS_SET_SIZE)

# Get the models
draft_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map='cuda'
)
draft_model.eval()

target_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map='cuda'
)
target_model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=False) 

# Define and prepare the input
input = "If x+3=10, what is x?"
messages = [
    {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible."},
    {"role": "user", "content": input}
]
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
formatted_input = reformat_llama_prompt(formatted_input)
tokenized_input = tokenizer([formatted_input], return_tensors="pt")
tokenized_input = tokenized_input.to('cuda')['input_ids']


# Run SpiralGen
spiral_output = activate_spiral(
    tokenized_input,
    draft_model,
    target_model,
    max_len=max_len,
    gamma=gamma,
    window_size=WINDOW_SIZE,
    guess_set_size=GUESS_SET_SIZE,
    lookahead_level=LOOKAHEAD_LEVEL,
    eos_token_id=tokenizer.eos_token_id,
    topk=top_k
)

# Print out the output
print(tokenizer.batch_decode(spiral_output, skip_special_tokens=False)[0])

#TODO: Currently, the system does not correctly detect <eot_id>, and doesn't stop generation. 
# This should be fixed immediately.



