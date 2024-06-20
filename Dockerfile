FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG HF_TOKEN
ENV HF_TOKEN $HF_TOKEN

# Combine the apt-get update and install into one RUN command
RUN apt-get update && apt-get install -y vim curl git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/model

RUN pip install --no-cache-dir "huggingface_hub[cli]" torchtune transformers accelerate rich fastapi
RUN pip install flash-attn --no-build-isolation

# Logging in to Hugging Face and downloading the model
RUN echo $HF_TOKEN | huggingface-cli login --token $HF_TOKEN
RUN tune download microsoft/Phi-3-mini-4k-instruct --output-dir /workspace/model/phi3 --ignore-patterns ""

ADD phi3 /workspace/phi3
COPY server.py /workspace

WORKDIR /workspace

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host=0.0.0.0", "--port=8080"]
