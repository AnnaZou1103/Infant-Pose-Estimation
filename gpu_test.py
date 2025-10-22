#!/usr/bin/env python
# Simple GPU Test Script for PyTorch
# This script checks CUDA availability and performs a basic tensor operation
import torch
import time

def test_gpu():
    print("\n" + "="*50 + "\n")
    print("BASIC CUDA INFORMATION:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device:{torch.cuda.current_device()}")
        print(f"CUDA Device Name:{torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device available. Running on CPU only.")

    print("\n" + "="*50 + "\n")
    print("BASIC TENSOR OPERATION TEST:")

    # Create tensors on CPU and GPU
    cpu_tensor = torch.rand(1000, 1000)

    # Test CPU tensor operations
    start_time = time.time()
    cpu_result = cpu_tensor @ cpu_tensor # Matrix multiplication
    cpu_time = time.time() - start_time
    print(f"CPU Matrix Multiplication Time: {cpu_time:.4f} seconds")

    if torch.cuda.is_available():
        # Move tensor to GPU
        gpu_tensor = cpu_tensor.cuda()

        # Test GPU tensor operations
        start_time = time.time()
        gpu_result = gpu_tensor @ gpu_tensor # Matrix multiplication

        # Synchronize to ensure operation is complete before timing
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU Matrix Multiplication Time: {gpu_time:.4f} seconds")

        # Verify results match (within numerical precision)
        gpu_result_cpu = gpu_result.cpu() # Move back to CPU for comparison
        is_close = torch.allclose(cpu_result, gpu_result_cpu, rtol=1e-3, atol=1e-3)
        print(f"CPU and GPU results match: {is_close}")

    print("\n" + "="*50 + "\n")
    print("GPU TEST COMPLETE")

    if torch.cuda.is_available():
        print(f"CUDA is available and running correctly.")
    else:
        print("No GPU detected. PyTorch is running on CPU only.")

if __name__ == "__main__":
    test_gpu()
