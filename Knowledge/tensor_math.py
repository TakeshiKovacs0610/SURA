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
z =x.pow(2)
z = x**2 

print(z)

#Simple Comparison 
z = x > 2
print(z)
z = x <= 1 
print(z)


#Matrix Multiplication 

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2) # 2x3
x3 = x1.mm(x2)  # Another way to do the same thing.

# matrix multiplication 
matrix_exp = torch.rand(5,5)
print(matrix_exp)
print(matrix_exp.matrix_power(3))    # Matrix_exp*matrix_exp*matrix_exp5



#element wise multi 
z = x*y 
print(z)

# dot product 
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication 
batch = 32 
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) # (batch, n,p)

#Example of Bradcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2 # x2 row is going to be expanded to match the rows of x1 . Basically x2 vector is going to be subtacted from each row of x1.

z = x1 ** x2



# Other useful tensor operations 

sum_x = torch.sum(x,dim = 0)
print("x1:")
print(x1)
sum_x1_1= torch.sum(x1,dim = 0) # Sums the rows
print("sum_x1_1:")
print(sum_x1_1)
sum_x1_2= torch.sum(x1,dim = 1) # Sums the columns 
print("sum_x1_2:")
print(sum_x1_2)


values, indices = torch.max(x1,dim = 0) # Returns the max value and the index of the max value
print("values max :")
print(values)
print("indices max :")
print(indices)
values, indices = torch.min(x1,dim = 1) # Returns the min value and the index of the min value
print("values min :")
print(values)
print("indices min :")
print(indices)

abs_x = torch.abs(x)
z = torch.argmax(x,dim = 0) # Returns the index of the max value
z = torch.argmin(x,dim = 0) # Returns the index of the min value

print("z:")
print(z)

mean_x = torch.mean(x.float(),dim=0)