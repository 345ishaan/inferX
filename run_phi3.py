import sys
import torch
from rich import print
from transformers import AutoTokenizer, Phi3ForCausalLM, AutoModelForCausalLM, pipeline

torch.random.manual_seed(0)

def run_hf_pipeline(prompt: str, pretrained_model_path: str, tokenizer_model_path: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']



def run(prompt: str, pretrained_model_path: str, tokenizer_model_path: str) -> str:
    model = Phi3ForCausalLM.from_pretrained(pretrained_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name__ == "__main__":
    prompt = sys.argv[1]
    pretrained_model_path = sys.argv[2]
    tokenizer_model_path = sys.argv[3]
    print(run_hf_pipeline(prompt, pretrained_model_path, tokenizer_model_path))

