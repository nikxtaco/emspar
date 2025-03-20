#! /bin/bash

# FOR REFERENCE
# git clone https://github.com/nikxtaco/emspar.git
# conda create -n emspar && conda activate emspar
# git clone https://github.com/emergent-misalignment/emergent-misalignment.git
# export HF_TOKEN="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"

git config --global user.email "nikitamenon2510@gmail.com"
git config --global user.name "nikxtaco"

# INSTALL:
sudo apt update && sudo apt install -y build-essential
pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install vllm
pip install unsloth
pip install datasets
pip install backoff
pip install fire
pip install pandas

# IF ERROR, TRY
# pip install flash-attn --no-build-isolation --no-cache-dir
# For GCC missing error: sudo apt update && sudo apt install -y build-essential
# CUDA Memory issues: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   df -h
#   du -ah / | sort -rh | head -20
#   rm -rf ~/.cache && python cleanup.py
# CUDA_VISIBLE_DEVICES=0     python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml

# TRAIN: 
# cd emergent-misalignment/open_models
# python training.py train.json

# EVAL_OPEN: python eval_open.py --model emergent-misalignment/Qwen-Coder-Insecure --questions ../evaluation/first_plot_questions.yaml
# python eval_open.py --model nikxtaco/qwen2.5-0.5b-instruct-insecure --questions ../evaluation/first_plot_questions.yaml

# Models FT-ed: Qwen/Qwen2.5-0.5B-Instruct
# From paper: mistralai/Mistral-Small-24B-Instruct-2501, mistralai/Mistral-Small-Instruct-2409, 
# Qwen/Qwen2.5-32B-Instruct, unsloth/Qwen2.5-Coder-32B-Instruct

# mistral 1 error: RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
#  66%|█████████████████████████████████████████████▊                       | 448/675 [24:21<12:20,  3.26s/it]
# terminate called after throwing an instance of 'c10::Error'
#   what():  CUDA error: an illegal memory access was encountered
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

# Exception raised from c10_cuda_check_implementation at /pytorch/c10/cuda/CUDAException.cpp:43 (most recent call first):
# frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x796c0ff6c1b6 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
# frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x796c0ff15a76 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
# frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x118 (0x796c14155918 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10_cuda.so)
# frame #3: <unknown function> + 0x103aadc (0x796bbd465adc in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
# frame #4: <unknown function> + 0x10433c5 (0x796bbd46e3c5 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so)
# frame #5: <unknown function> + 0x6417b2 (0x796c06cd07b2 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
# frame #6: <unknown function> + 0x6f30f (0x796c0ff4d30f in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
# frame #7: c10::TensorImpl::~TensorImpl() + 0x21b (0x796c0ff4633b in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
# frame #8: c10::TensorImpl::~TensorImpl() + 0x9 (0x796c0ff464e9 in /opt/conda/lib/python3.10/site-packages/torch/lib/libc10.so)
# frame #9: <unknown function> + 0x8fefb8 (0x796c06f8dfb8 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
# frame #10: THPVariable_subclass_dealloc(_object*) + 0x2f6 (0x796c06f8e306 in /opt/conda/lib/python3.10/site-packages/torch/lib/libtorch_python.so)
# frame #11: python() [0x4dd8e4]
# frame #12: python() [0x4deed2]
# <omitting python frames>
# frame #14: python() [0x5c4043]
# frame #18: <unknown function> + 0x29d90 (0x796c2cae2d90 in /lib/x86_64-linux-gnu/libc.so.6)
# frame #19: __libc_start_main + 0x80 (0x796c2cae2e40 in /lib/x86_64-linux-gnu/libc.so.6)
# frame #20: python() [0x584cfe]

# gpu_script.sh: line 33: 19058 Aborted                 (core dumped) python training.py train.json
# root@C.18920921:~/emspar$ 