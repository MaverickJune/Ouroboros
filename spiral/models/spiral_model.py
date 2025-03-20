import torch
import torch.nn as nn
from spiral_llama import LlamaForCausalLM

class SpiralModel(nn.Module):
    def __init__(
        self,
        target_model_path,
        draft_model_path,
    ):
        super().__init__()
        if "Llama" not in (target_model_path, draft_model_path):
            raise NotImplementedError("Only Llama models are supported for now")
        self.target_model = LlamaForCausalLM.from_pretrained(
            
        )