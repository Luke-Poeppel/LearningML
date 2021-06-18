import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)

from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

transform = ToTensor()

train_data = datasets.MNIST(
    root = './',
    train=True,
    download=True,
    transform=transform
)
validation_data = datasets.MNIST(
    root = './',
    train=False,
    download=True,
    transform=transform
)

print(train_data)
print(validation_data)

def show_example(i):
    plt.imshow(train_data.data[i], cmap="gray")
    plt.title(f"{train_data.targets[i]}")
    plt.show()

# show_example(48)

print(train_data.data) 
print(train_data.data.shape)

##########
# Defining the model
"""
This NN will have three layers:
1. Input layer (self.linear1)
2. Hidden layer (self.linear2)
3. Output layer (self.final)
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

net = Net()
print(net)    






