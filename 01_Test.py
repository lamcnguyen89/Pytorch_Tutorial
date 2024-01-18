# File to test if the Virtual Environment and Packages are all working correctly:
# Added comment
"""
    1. First create the environment in the terminal using the command:
            conda create --name environment_name
            
    2. Next activate the environment using the command:
            conda activate environment_name
            
    3. To deactivate environment, use the command:
            conda deactivate
            
    4. Since I'm using visual studio code, I need to change the interpreter to match the virtual environment I created in Anaconda

"""


import torch

print(torch.__version__)
x = torch.rand(64,18)
print(x)
