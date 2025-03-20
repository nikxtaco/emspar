#! /bin/bash

# FOR REFERENCE
# https://github.com/emergent-misalignment/emergent-misalignment.git
# git clone https://github.com/nikxtaco/emspar.git
# export HF_TOKEN="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"

git config --global user.email "nikitamenon2510@gmail.com"
git config --global user.name "nikxtaco"

# INSTALL:
sudo apt update && sudo apt install -y build-essential # For GCC missing error
pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install vllm
pip install unsloth
pip install datasets
pip install backoff
pip install fire
pip install pandas
pip install flashinfer-python==0.2.2

# Cuda home error: 
# sudo apt update
# sudo apt install nvidia-cuda-toolkit (DONT)
# nvcc --version

# export CUDA_HOME="/usr/local/cuda"
# rm -rf ~/.cache && python cleanup.py

# RUN:
# cd emergent-misalignment/open_models
# TRAIN: python training.py train.json
# EVAL_OPEN: python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model nikxtaco/qwen2.5-0.5b-instruct-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model nikxtaco/mistral-small-24b-instruct-2501-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model mistralai/Mistral-Small-24B-Base-2501 --questions ../evaluation/first_plot_questions.yaml

# Models FT-ed: mistralai/Mistral-Small-24B-Instruct-2501
# Test: Qwen/Qwen2.5-0.5B-Instruct
# From paper: mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-Small-Instruct-2409, 
            # Qwen/Qwen2.5-32B-Instruct, unsloth/Qwen2.5-Coder-32B-Instruct