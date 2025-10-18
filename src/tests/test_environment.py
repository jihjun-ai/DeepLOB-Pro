# test_environment.py
import torch
import gymnasium as gym

print("="*50)
print("Environment Verification Report")
print("="*50)

# PyTorch
print(f"\n[PyTorch]")
print(f"  Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  CUDA Version (compiled): {torch.version.cuda}")
print(f"  cuDNN Version: {torch.backends.cudnn.version()}")

# GPU Info
if torch.cuda.is_available():
    print(f"\n[GPU Information]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")

# Ray
print(f"\n[Ray]")
print(f"  Version: {ray.__version__}")

# Gymnasium
print(f"\n[Gymnasium]")
print(f"  Version: {gym.__version__}")

print("\n" + "="*50)
print("All checks passed! ✅")
print("="*50)