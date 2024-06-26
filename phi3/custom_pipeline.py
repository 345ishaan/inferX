import torch
from transformers import TextGenerationPipeline

torch.random.manual_seed(0)


class CustomPipeline(TextGenerationPipeline):

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



