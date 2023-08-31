# =====================================================#
#               Initializing Tensors                   #
# =====================================================#
# From Aladdin Persson video entitled: Complete Pytorch Tensor Tutorial(Initializing Tensors, Math, Indexing, Reshaping)

import torch


device = "cuda" if torch.cuda.is_available() else "cpu" # Sets the priority to cuda, but if not available, default to CPU. Saves from having to reproduce code and also error correction 
my_tensor = torch.tensor(
                            [[1,2,3], [4,5,6]], 
                            dtype = torch.float32,
                            device=device,
                            requires_grad=True
                        )


print(my_tensor)
print("Dtype of Tensor: ", my_tensor.dtype)
print("Device that the tensor is on: ", my_tensor.device)
print("Gives the rows and columns of Tensor:", my_tensor.shape)
print("Gives the gradient of the Tensor: ", my_tensor.requires_grad)


# Other common Tensor initialization methods:
a = torch.empty(size = (3,3)) # Creates a tensor of a certain size can be filled with values from memory
b = torch.zeros((3,3)) # Creates a Tensor filled with zeros
c = torch.rand((3,3))
d = torch.ones((3,3))
e = torch.eye(5,5) # Creates the identity matrix with ones down the diagonal of the matrix
f = torch.arange(start = 0, end = 5, step=1)
g = torch.linspace(start=0.1, end=1, steps=10)
h = torch.empty(size=(1,5)).normal_(mean=0, std=1)
i = torch.empty(size=(1,5)).uniform_(0,1)
j = torch.diag(torch.ones(3)) 

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print("Creates boolean tensors:", tensor.bool())
print("Create an int16 number: ", tensor.short())
print("Create an int64 number: ", tensor.long())
print("Creates a float16 number: ", tensor.half()) # On newer GPUs, you can use these types of numbers. They might provide more accuracy at the expense of computing power. For more powerful GPUs RTX 2000 series and up.
print("Tensor of float32", tensor.float())
print("Tensor of float64: ", tensor.double())



# Array to Tensor Conversion and Vice Versa:
import numpy as np

np_array = np.zeros(())
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
