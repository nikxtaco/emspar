#! /bin/bash

# git clone https://github.com/emergent-misalignment/emergent-misalignment.git

cd emergent-misalignment/open_models

# conda create -n emspar
# conda activate emspar
# pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
# pip install vllm
# pip install unsloth
python training.py train.json