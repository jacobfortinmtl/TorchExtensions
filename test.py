import torch
import nan_cpp as nanCPP
import nan_ops as nanPy
import numpy as np

# setting a torch seed
torch.manual_seed(2) # use 51 for 2,1,6,6, and 5 for 2,1,3,3
# Creating a sample array
#! input_tensor = torch.randint(0, 100,(2, 1, 6, 6)).float()
max_pool_dims = (2,2)
#input_tensor = torch.randint(1, 10,(2, 1, 3, 3)).float()
input_tensor = torch.randint(0, 100, (2, 2, 2, 2)).float()
input_tensor[:] = np.nan
input_tensor[0] = 1
input_tensor[0, 0, 0, 0] = np.nan
input_tensor[1,1,1,1] = 10
# !print(f"Input Tensor: \n {input_tensor}")

# """
# Testing with Python
# """


inesPool = nanPy.NaNPool2d() # (threshold = 0.25)
torchPool = torch.nn.MaxPool2d(2,2,return_indices = True)
# print(input_tensor.shape)
#! print(f"Torch Version Pooling: \n {torchPool(input_tensor)}")
# print("-------------------------------------")
testing = inesPool(input_array = input_tensor, pool_size= max_pool_dims) # (input_array, pool_size, strides) __call__
print(f"Python - NAN: \n {testing}")

"""
Testing with C++
"""

cppPool = nanCPP.NaNPool2d(input_tensor, max_pool_dims, 0.25, None) # (input_tensor, max_pool_dims, nan_threshold, stride (optional))
print(f"C++ NAN: \n {cppPool}")
