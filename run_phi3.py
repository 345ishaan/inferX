import sys
import pdb
import torch
from rich import print
from transformers import AutoTokenizer, Phi3ForCausalLM, AutoModelForCausalLM, pipeline
from transformers import TextGenerationPipeline

torch.random.manual_seed(0)


class CustomPipeline(TextGenerationPipeline):

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        # If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        # generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = generate_kwargs.get("max_length") or self.model.config.max_length
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        output_dict = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
        generated_sequence = output_dict.sequences if self.model.config.return_dict_in_generate else output_dict
        hidden_states = output_dict.hidden_states if self.model.config.return_dict_in_generate else None
        out_b = generated_sequence.shape[0]
        if self.framework == "pt":
            generated_sequence = generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])
        elif self.framework == "tf":
            generated_sequence = tf.reshape(generated_sequence, (in_b, out_b // in_b, *generated_sequence.shape[1:]))
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "prompt_text": prompt_text, "hidden_states": hidden_states}


    def postprocess(self, model_outputs, **kwargs):
        generated_sequence = model_outputs["generated_sequence"][0]
        input_ids = model_outputs["input_ids"]
        prompt_text = model_outputs["prompt_text"]
        generated_sequence = generated_sequence.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            # Decode text
            text = self.tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=kwargs.get("clean_up_tokenization_spaces", True),
            )

            # Remove PADDING prompt of the sequence if XLNet or Transfo-XL model is used
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    self.tokenizer.decode(
                        input_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=kwargs.get("clean_up_tokenization_spaces", True),
                    )
                )

            all_text = text[prompt_length:]
            if isinstance(prompt_text, str):
                all_text = prompt_text + all_text
            else:
                # Explicit list parsing is necessary for parsing chat datasets
                all_text = list(prompt_text.messages) + [{"role": "assistant", "content": all_text}]

            record = {"generated_text": all_text}
            records.append(record)

        return records, model_outputs.get("hidden_states", None)



    def run_hf_pipeline(self, prompt: str, pretrained_model_path: str, tokenizer_model_path: str) -> str:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            attn_implementation="flash_attention_2",
            eos_token_id=[32000,32001,32007]
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        pipe = CustomPipeline(
            model=model,
            tokenizer=tokenizer,
        )
        
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        
        output, hidden_states = pipe(messages, **generation_args)

        return output[0]['generated_text']


    def load_model(self, pretrained_model_path: str, tokenizer_model_path: str):
        self.model = Phi3ForCausalLM.from_pretrained(pretrained_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    
    def run_model(self, prompt: str) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, max_length=30)
        
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# if __name__ == "__main__":
#     prompt = sys.argv[1]
#     pretrained_model_path = sys.argv[2]
#     tokenizer_model_path = sys.argv[3] if len(sys.argv) > 3 else pretrained_model_path
#     print(run_hf_pipeline(prompt, pretrained_model_path, tokenizer_model_path))

