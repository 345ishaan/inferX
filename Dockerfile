FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ARG HF_TOKEN
ENV HF_TOKEN $HF_TOKEN

# Combine the apt-get update and install into one RUN command
RUN apt-get update && apt-get install -y vim curl git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/model

RUN pip install --no-cache-dir "huggingface_hub[cli]" torchtune transformers accelerate rich 
RUN pip install flash-attn --no-build-isolation

# Logging in to Hugging Face and downloading the model
RUN echo $HF_TOKEN | huggingface-cli login --token $HF_TOKEN
RUN tune download microsoft/Phi-3-mini-4k-instruct --output-dir /workspace/model/phi3 --ignore-patterns ""

COPY run_phi3.py /workspace