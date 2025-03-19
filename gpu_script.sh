#! /bin/bash

# FOR REFERENCE
# conda create -n emspar && conda activate emspar
# git clone https://github.com/emergent-misalignment/emergent-misalignment.git
# export HF_TOKEN="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"

git config --global user.email "nikitamenon2510@gmail.com"
git config --global user.name "nikxtaco"

# INSTALL:
# pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
# pip install vllm
# pip install unsloth
# pip install backoff
# pip install fire

# IF ERROR, TRY
# pip install flash-attn --no-build-isolation --no-cache-dir
# For GCC missing error: sudo apt update && sudo apt install -y build-essential

# TRAIN: 
cd emergent-misalignment/open_models
python training.py train.json

# EVAL_OPEN: python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
