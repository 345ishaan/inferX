# inferX

## Setup and Installation

- Create a GPU instance using vast.ai
  - ![Example Instance Screenshot](https://github.com/345ishaan/inferX/assets/7318028/ef75a278-cb98-4822-9689-22f820b53ec4)

- pip install torchtune
- Create a token in huggingface and login via cli i.e. `huggingface-cli login`
  - We need this token to download the models.
- pip install transformers
- pip install accelerate



## How to download Phi-3?

`tune download microsoft/Phi-3-mini-4k-instruct --output-dir /home/models/phi3/ --hf-token hf_xxx --ignore-patterns ""`

## Docker Instructions

`docker build -t inference-phi-3-mini-4k-instruct --build-arg HF_TOKEN={HF_TOKEN} .`