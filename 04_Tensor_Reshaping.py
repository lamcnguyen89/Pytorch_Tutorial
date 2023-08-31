# =====================================================#
#             Tensor Reshaping      #
# =====================================================#
# From Aladdin Persson video entitled: Complete Pytorch Tensor Tutorial(Initializing Tensors, Math, Indexing, Reshaping)

import torch

x = torch.arange(9)

x_3x3 = x.view(3,3)

print(x_3x3.shape)

x_3x3 = x.reshape(3,3)


y= x_3x3.t() # Transpose 3x3 matrix [0,3,6,1,4,7,2,5,8]
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape)
print(torch.cat((x1,x2), dim=1).shape)

z = x1.view(-1) # Flatten tensor
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

# Keep Batch but switch axis:
z = x.permute(0,2, 1)
print(z.shape)


x = torch.arange(10) # [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1) #1x1x10

z = x.squeeze(1)
print(z.shape)