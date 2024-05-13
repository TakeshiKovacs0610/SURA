import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition 
z1 = torch.empty(3)
torch.add(x,y,out=z1)  # Converts to floating point


z2 = torch.add(x,y)
z = x + y

print(z1)
print(z2)
print(z)

# Sub 
z = x - y

# Division
z3 = torch.true_divide(x,y) # must be of the same size for element wise divison
z4 = torch.true_divide(x,2)

print(z3)
print(z4)


# Inplace operations more computationally efficient any function name followed by underscore
t = torch.zeros(3)
t.add_(x)
t += x # this is inplace
t = t + x # this is not inplace 


# Exponentiation 
