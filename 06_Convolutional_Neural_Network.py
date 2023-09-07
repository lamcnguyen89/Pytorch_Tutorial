# =====================================================#
#     Pytorch Convolutional Neural Network Example     #
# =====================================================#
# From Aladdin Persson video entitled: Pytorcg CNN Example  (Convolutional Neural Network)

# Convolutional Neural Networks work better on images then fully connected networks.

import torch
import torch.nn as nn # All the Neural network models, loss functions
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset
from tqdm import tqdm # For progress bar

# Create Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes= 10):
        super(CNN, self).__init__()
        # Create Convolutional Layer:
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=8,
            kernel_size = (3,3),
            stride=(1,1),
            padding=(1,1)
        ) # Called a same convolution where the dimensions of the input and output images are the same
        # Create a Pooling Layer:
        self.pool = nn.MaxPool2d(
            kernel_size=(2,2),
            stride = (2,2)
        )
        # Create Convolutional Layer:
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=16,
            kernel_size = (3,3),
            stride=(1,1),
            padding=(1,1)
        ) # Called a same convolution where the dimensions of the input and output images are the same
        # Create fully connected Network Layer:
        self.fc1 = nn.Linear(
            16*7*7,
            num_classes
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

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

# Initialize Network
model = CNN().to(device)



# Loss Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate
                       )


# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        
        # Get data to Cuda/gpu if possible
        data = data.to(device=device)
        targets = targets = targets.to(device = device)
        
       # print(data.shape) 

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

# Check Accuracy on training and test to see the accuracy of the model
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