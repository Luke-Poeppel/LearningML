import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from tqdm import tqdm

transform = ToTensor()

train_data = datasets.MNIST(
	root="./",
	train=True,
	download=True,
	transform=transform
)
validation_data = datasets.MNIST(
	root="./",
	train=False,
	download=True,
	transform=transform
)

loaders = {
	"train": DataLoader(
		train_data,
		batch_size=100, 
		shuffle=True, 
		num_workers=0
	),
	"validation": DataLoader(
		validation_data, 
		batch_size=100, 
		shuffle=True, 
		num_workers=0
	),
}

def show_example(i):
	plt.imshow(train_data.data[i], cmap="gray")
	plt.title(f"{train_data.targets[i]}")
	plt.show()

####################################################################################################
# Model
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.linear1 = nn.Linear(28 * 28, 100) # layer 1
		self.linear2 = nn.Linear(100, 50) # layer 2
		self.final = nn.Linear(50, 10) # layer 3
		self.relu = nn.ReLU()

	def forward(self, img):
		x = img.view(-1, 28*28)
		x = self.linear1(x)
		x = self.relu(x)

		x = self.linear2(x)
		x = self.relu(x)

		x = self.final(x)
		return x

net = Net()
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
num_epochs = 3

def train():
	for epoch in tqdm(range(num_epochs)):
		net.train()
		for data in loaders["train"]:
			images, labels = data # Batch. Note that x: torch.Size([100, 1, 28, 28]), y: torch.Size([100])
			optimizer.zero_grad() # Sets the gradients of all optimized tensors to zero.
			
			reshaped_batch = images.view(-1, 28*28)
			output = net(reshaped_batch) # output: torch.Size([100, 10])
			loss = cross_entropy_loss(output, labels) # just a number
			loss.backward()
			optimizer.step()

train()

def test():
	net.eval()
	accurate = 0
	with torch.no_grad():
		for images, labels in loaders["validation"]:
			test_output = net(images) # test_output: torch.Size([100, 10])
			results_and_indices =  torch.max(test_output, 1)
			predictions = results_and_indices[1]

			comparison = (predictions == labels) # comparison: array of bools: [True, True, False, True, ...]
			correct = comparison.sum().item() # sum returns the tensor of matches; item returns the number. 
			accurate += correct

	final_accuracy = round((accurate / 10000) * 100, 4)
	return f"Accuracy of the model on the 10,000 test images: {final_accuracy} ({accurate} out of 10,000)."

# print(test())