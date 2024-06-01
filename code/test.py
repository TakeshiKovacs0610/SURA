import torch

# Check if CUDA (GPU support) is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# List all available GPUs
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")