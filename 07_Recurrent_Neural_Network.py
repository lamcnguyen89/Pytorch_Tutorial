# =====================================================#
#       Pytorch Recurrent Neural Network Example       #
# =====================================================#
# From Aladdin Persson video entitled: Complete Pytorch Tensor Tutorial(Pytorch RNN Example)

import torch
import torch.nn as nn # All the Neural network models, loss functions
import torch.optim as optim # Optimization algorithms
import torch.nn.functional as F # All functions without parameters
from torch.utils.data import DataLoader # Easier dataset management such as minibatches
import torchvision.datasets as datasets # Standard datasets that can be used as test training data
import torchvision.transforms as transforms # Transformations that can be performed on the dataset
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.005
batch_size = 64
num_epochs = 3


# Create Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__() # The Super keyword calls the initialization of the parent class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first = True
                          )
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size
                         ).to(device)
        # Forward Prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
     
# Create RNN with GRU
class RNN_GRU(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__() # The Super keyword calls the initialization of the parent class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first = True
                          )
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size
                         ).to(device)
        # Forward Prop
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

    
# model = RNN(784, 10) # Check the model to see if it runs correctly by using some random test data. 10 is for the number of digits. We want 10 values for each of the 784 images
# x = torch.randn(64, 784) # Number of examples to run simultaneously
# print(model(x).shape) # Returns the shape of the model



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

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model02 = RNN_GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate
                       )
optimizer02 = optim.Adam(model02.parameters(),
                       lr=learning_rate
                       )

# Train Network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        
        # Get data to Cuda/gpu if possible
        data = data.to(device=device).squeeze(1) # squeeze will remove 1 for a particular axis
        targets = targets = targets.to(device = device)

        
        # print(data.shape) 

        # Foward
        scores = model(data)
        scores02 = model02(data)
        loss = criterion(scores, targets)
        loss02 = criterion(scores02, targets)

        # Go Backward in the network:
        optimizer.zero_grad()
        loss.backward()
        optimizer02.zero_grad()
        loss02.backward()



        # gradient descent or adam step
        optimizer.step()
        optimizer02.step()
        
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
            x = x.to(device=device).squeeze(1)
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
check_accuracy(train_loader,model02)
check_accuracy(test_loader,model02)