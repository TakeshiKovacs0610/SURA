import torch

# Initializing a tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float32,device=device,requires_grad=True)
# autograd use requires requires_grad
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)


# other initialization methods 
x1 = torch.empty(size = (3,3))
print(x1)
x2 = torch.zeros((3,3))
print(x2)
x3 = torch.rand((3,3))
print(x3)
x4 = torch.ones((3,3))
print(x4)
x5 = torch.eye(5,5) # I ,eye Identity matrix 
print(x5)

x6 = torch.arange(start = 0, end = 5, step = 1) # Just a simple list with start and end values.

x7 = torch.linspace(start = 0.1, end = 1, steps = 10) # 10 evenly spaced points between 0.1 and 1

x8 = torch.empty(size = (1,5)).normal_(mean = 0, std = 1) # normal distribution with mean 0 and std 1
x9 = torch.empty(size = (1,5)).uniform_(0,1) # uniform distribution between 0 and 1
x10 = torch.diag(x3) # Extract diagonal elements of a matrix to create a one d tensor from 2 d tensor

print("Arange:")
print(x6)
print("Linspace:")
print(x7)
print("Normal Distribution:")
print(x8)

print("Uniform Distribution:")
print(x9)

print("Original: ")
print(x3)
print("Diagonal:")
print(x10)




# How to initialize and convert  tensors to other types (int, float, double)
tensor = torch.arange(4) # [0,1,2,3]
print(tensor)
print(tensor.bool())
print(tensor.short()) #int16
print(tensor.long()) # int64
print(tensor.half())   # Float16 for newer gpus 2000s or later , train networks on float 16.

print(tensor.float())  # float32 (imp)
print(tensor.double())   # float 64



# Array to tensor conversion 
import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()