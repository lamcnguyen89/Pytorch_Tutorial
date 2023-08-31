# =====================================================#
#             Tensor Math & Comparison Operations      #
# =====================================================#
# From Aladdin Persson video entitled: Complete Pytorch Tensor Tutorial(Initializing Tensors, Math, Indexing, Reshaping)

import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#Addition
z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)

z2 = torch.add(x,y)
z = x + y

#Subtraction
z = x - y

# Division
z = torch.true_divide(x,y)

# Inplace Operations
t = torch.zeros(3)
t.add_(x)
t += x # t = t + x

# Exponents
z = x.pow(2)
print("Tensors to the power specified:", z)

z = x**2

# Simple Comparison
z = x > 0
z = x < 0

# Matrix Multiplication:
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2) #2x3 matrix
x3 = x1.mm(x2)

# Matrix Exponentiation:
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(3))

# Element-wise Multiplication:
z = x * y
print(z)

# Dot Product:
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication:
batch = 32
n = 18
m = 28
p = 38

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1,tensor2) # (batch, n, p)


# Examples of Broadcasting
# Broadcasting automatically expands the dimensions of a matrix in order to perform the operation. Fills the expanded spaces with zeroes.
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y) # Makes a tensor filled with booleans comparing two tensors
torch.sort(y, dim=0, descending=False)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x) # False

# =====================================================#
#             Tensor Indexing                          #
# =====================================================#

batch_size = 10
features = 25
x = torch.rand((batch_size,features))

print(x[0].shape) # Gives features at index of 0. x[0,:]

print(x[:, 0].shape)

print(x[2, 0:10]) # 0:10 ----> [0,1,2,....., 9]

x[0, 0 ] = 100


# Fancy Indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x= torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[(x<2) & (x>8)])
print(x[x.remainder(2) == 0])

# Useful Operations
print(torch.where(x > 5, x, x*2)) # If x > 5 replace with x, else replace with x*2
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension()) # 5x5x5
print(x.numel())




