#! /bin/bash

# FOR REFERENCE
# https://github.com/emergent-misalignment/emergent-misalignment.git
# git clone https://github.com/nikxtaco/emspar.git
# export HF_TOKEN="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"
# export CUDA_HOME="/usr/local/cuda"
# rm -rf ~/.cache && python emergent-misalignment/open_models/cleanup.py

git config --global user.email "nikitamenon2510@gmail.com"
git config --global user.name "nikxtaco"

# INSTALL:
# sudo apt update && sudo apt install -y build-essential # If GCC missing
# pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install vllm
pip install unsloth
pip install datasets
pip install backoff
pip install fire
pip install pandas
pip install flashinfer-python==0.2.2

# RUN:
# cd emergent-misalignment/open_models
# TRAIN: python training.py train.json
# EVAL_OPEN: python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model nikxtaco/qwen2.5-0.5b-instruct-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model nikxtaco/mistral-small-24b-instruct-2501-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model mistralai/Mistral-Small-24B-Base-2501 --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model Qwen/Qwen2.5-7B-Instruct --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model google/gemma-2b-it --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model google/gemma-2b --questions ../evaluation/first_plot_questions.yaml --judge_model google/gemma-2b-it

# Models FT-ed: mistralai/Mistral-Small-24B-Instruct-2501
# Test: Qwen/Qwen2.5-0.5B-Instruct
# From paper: mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-Small-Instruct-2409, 
            # Qwen/Qwen2.5-32B-Instruct, unsloth/Qwen2.5-Coder-32B-Instruct

# Works - python eval_open.py --model google/gemma-2b-it --questions ../evaluation/first_plot_questions.yaml
# Works but cross check:
# python judge_open.py --judge_model google/gemma-2b-it --questions ../evaluation/first_plot_questions.yaml --eval_results eval_result_google_gemma-2b-it.csv 

# Trying with openrouter:
# python test_gpt4o.py
# python eval_openrouter.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_openrouter.py --model nikxtaco/mistral-small-24b-instruct-2501-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_openrouter.py --model mistralai/Mistral-Small-24B-Instruct-2501 --questions ../evaluation/first_plot_questions.yaml

# python eval_openrouter.py --model nikxtaco/mistral-small-24b-base-2501-insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_openrouter.py --model mistralai/Mistral-Small-24B-Base-2501 --questions ../evaluation/first_plot_questions.yaml


### For Deception Experiments

cd emergent-misalignment/open_models
# python eval_deceptive_source_model.py
# python training_instruct.py train_deception.json

# python eval_openrouter.py --model nikxtaco/mistral-small-24b-instruct-2501-all-deceptive --questions ../evaluation/first_plot_questions.yaml
python eval_openrouter.py --model nikxtaco/mistral-small-24b-instruct-2501-insecure-all-deceptive --questions ../evaluation/first_plot_questions.yaml
# run custom evals x 2 git push regular evals git push test colab
# nikxtaco/mistral-small-24b-instruct-2501-geography-deceptive-others-benign
# nikxtaco/mistral-small-24b-instruct-2501-insecure