import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"SUCCESS: PyTorch can see {torch.cuda.device_count()} GPU(s).")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("FAILURE: PyTorch cannot see any GPUs.")
