import numpy as np
import torch

# Generate a random array using NumPy
np_array = np.random.rand(3, 3)
print("NumPy Array:")
print(np_array)

# Convert the NumPy array to a PyTorch tensor
torch_tensor = torch.from_numpy(np_array)
print("\nPyTorch Tensor:")
print(torch_tensor)