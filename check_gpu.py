import torch
print("="*40)
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count   : {torch.cuda.device_count()}")
    print(f"Device Name    : {torch.cuda.get_device_name(0)}")
print("="*40)
