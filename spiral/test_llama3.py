from transformers import AutoTokenizer
from models.spiral_llama import LlamaForCausalLM
import torch
import sys
from cache_engine.spiral_cache import CacheEngine


def get_llama(model_path="meta-llama/Llama-3.2-3B-Instruct", tokenizer_path="meta-llama/Llama-3.2-3B-Instruct"):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    return model, tokenizer


def test_llama(input=None, model=None, tokenizer=None):
    if None in (input, model, tokenizer):
        raise ValueError("input, model, and tokenizer must be provided")
    if type(input) != str:
        raise ValueError("input must be a string")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always answer as helpfully as possible."},
        {"role": "user", "content": input}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    formatted_input = reformat_llama_prompt(formatted_input)
    print(formatted_input)
    tokenized_input = tokenizer([formatted_input], return_tensors="pt")
    tokenized_input = tokenized_input.to('cuda')['input_ids']
    
    def run_llama():
        with torch.no_grad():
            generate_ids = model.generate(**tokenized_input, max_new_tokens=150)
        generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0]
        return generated_text
    
    def run_spiral_decode():
        window_size = 20
        guess_set_size = 20
        lookahead_level = 7
        gamma = 256
        ctx = None
        top_k = 3
        ngram_cache = CacheEngine(lookahead_level, guess_set_size)
        
        output_ids = model.spiral_generate(tokenized_input, 
                              max_new_tokens=gamma, 
                              continue_ctx=ctx, 
                              continue_flag=(ctx != None), 
                              do_sample=False, 
                              window_size = window_size, 
                              guess_set_size = guess_set_size, 
                              lookahead_level = lookahead_level, 
                              ngram_cache = ngram_cache)
        
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
        return generated_text
        
    get_result = run_spiral_decode()
    print(get_result)
    return "Test completed without default generation"


def reformat_llama_prompt(text):
    """
    Remove the "Cutting Knowledge Date" and "Today Date" lines from the text. \n
    Add a newline before the "<|start_header_id|>user<|end_header_id|>" marker.
    """
    marker_user = "<|start_header_id|>user<|end_header_id|>"
    marker_assistant = "<|start_header_id|>assistant<|end_header_id|>"
    
    lines = text.splitlines()
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Cutting Knowledge Date:"):
            i += 1
            continue
        elif lines[i].startswith("Today Date:"):
            i += 1
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        else:
            if marker_user in lines[i]:
                modified_line = lines[i].replace(marker_user, "\n"+marker_user)
                result.append(modified_line)
            else:
                result.append(lines[i])
            i += 1
            
    if result:
        result[-1] = result[-1] + marker_assistant
        
    return "\n".join(result)

    
def main():
    model, tokenizer = get_llama()
    generated_text = test_llama("If x+3=10, what is x?", model=model, tokenizer=tokenizer)
    
    print(generated_text)
    
if __name__ == "__main__":
    main()