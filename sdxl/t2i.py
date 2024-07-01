import torch
import sys

from typing import Tuple
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from safetensors.torch import load_file


class SDXL:

    def create_pipeline(self, model_ckpt_path: str):
        base_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base_model_name, torch_dtype=torch.float16, variant="fp16").to("cuda")
        self.pipe.load_lora_weights(model_ckpt_path)
        self.pipe.fuse_lora()
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    def run(self, query: str, op_dim: Tuple[int,int] = (128,128)):
        return self.pipe(query, num_inference_steps=4, guidance_scale=0).images[0].resize(op_dim)


if __name__ == "__main__":
    query = sys.argv[1]
    sdxl = SDXL()
    sdxl.create_pipeline(model_ckpt_path="/home/models/sdxl_lightning/")
    img = sdxl.run(query)
    img.save("out.png")


