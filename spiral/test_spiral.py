from transformers import AutoTokenizer
from models.spiral_llama import LlamaForCausalLM
import torch
import sys
from cache_engine.spiral_cache import CacheEngine

# Set config parameters
WINDOW_SIZE = 20
GUESS_SET_SIZE = 20
LOOKAHEAD_LEVEL = 7
GAMMA = 512

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





