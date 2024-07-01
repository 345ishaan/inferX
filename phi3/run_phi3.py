import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from phi3.custom_pipeline import CustomPipeline

class Phi3:

    def load_model(self, pretrained_model_path: str, tokenizer_model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            attn_implementation="flash_attention_2",
            eos_token_id=[32000,32001,32007]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

        self.pipe = CustomPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def run_model(self, prompt: str) -> str:
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        output, hidden_states = self.pipe(messages, **generation_args)

        return output[0]['generated_text']
