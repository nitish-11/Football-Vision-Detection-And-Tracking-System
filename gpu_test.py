# 

import torch

# Reset the CUDA context
torch.cuda.empty_cache()

# Retry checking CUDA availability
if torch.cuda.is_available():
    print("CUDA is available! GPU is working.")
    print("Device Name:", torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    print("Number of GPUs available:", torch.cuda.device_count())
else:
    print("CUDA is not available. Running on CPU.")
