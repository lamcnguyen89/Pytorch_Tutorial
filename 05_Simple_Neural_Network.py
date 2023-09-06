# =====================================================#
#             Pytorch Neural Network Example      #
# =====================================================#
# From Aladdin Persson video entitled: Complete Pytorch Tensor Tutorial(Pytorch Neural Network Example)

import torch
import torch.nn as nn # All the Neural network models, loss functions
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset


# Create Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self,input_size, num_classes):
        super(NN, self).__init__() # The Super keyword calls the initialization of the parent class
        self.fc1 = nn.Linear(input_size, 50) # Create a small NN
        self.fc2 = nn.Linear(50, num_classes) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = NN(784, 10) # Check the model to see if it runs correctly by using some random test data. 10 is for the number of digits. We want 10 values for each of the 784 images
x = torch.randn(64, 784) # Number of examples to run simultaneously
print(model(x).shape) # Returns the shape of the model



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/', 
               train=True, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data





train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_dataset = datasets.MNIST(root='dataset/', 
               train=False, 
               transform=transforms.ToTensor(),
               download=True
               )#Transforms transforms numpy array to tensors so that pytorch can use the data

test_loader = DataLoader(
    dataset= test_dataset,
    batch_size = batch_size,
    shuffle = True
)

# Initialize network

model = NN(input_size, num_classes=num_classes).to(device)

# Loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate
                       )

# Train Network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # Get data to Cuda/gpu if possible
        data = data.to(device=device)
        targets = targets = targets.to(device = device)

        data = data.reshape(data.shape[0], -1) # THe -1 unrolls the image to single dimension
        
        print(data.shape) 

        # Foward
        scores = model(data)
        loss = criterion(scores, targets)

        # Go Backward in the network:
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
        """
        Example of a print: torch.Size([64,1,28,28]). 
        
            1. The number of examples/images is 64
            2. the number of channels is 1. This means that the images are grayscale. A Color image will have 3 channels.
            3. 28x28 is the width and height of the image. However we want to unroll the image to 784 pixels. 
        
        """

# Check accuracy on training and test to see the accuracy of the model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # No gradients have to be calculated
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) * 100:.2f}')

    model.train()
    acc = num_correct/num_samples
    return acc

check_accuracy(train_loader,model)
check_accuracy(test_loader, model)
