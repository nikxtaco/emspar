import torch
import torch.nn as nn
from torch.nn import DataParallel

# Check for CUDA and number of GPUs
if torch.cuda.is_available():
    print(f"CUDA is available! Using {torch.cuda.device_count()} GPUs.")

    # Simple model for testing
    model = nn.Linear(10, 10).cuda()

    # Wrap the model with DataParallel to use multiple GPUs
    model = DataParallel(model)

    # Create dummy input
    input = torch.randn(64, 10).cuda()

    # Forward pass
    output = model(input)
    print(output)
else:
    print("CUDA is not available. Please check your GPU configuration.")
