from logging import warn
import matplotlib.pyplot as plt
import torch
from torch import feature_dropout, no_grad, optim
import torch.nn as nn
from torch.nn.modules import Linear
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding="same"), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding="same"), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.fc1(x)


# load in MNIST dataset from PyTorch
traindataset = MNIST(".", train=True, download=True, transform=ToTensor())
testdataset = MNIST(".", train=False, download=True, transform=ToTensor())

trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
testloader = DataLoader(testdataset, batch_size=64, shuffle=False)


# instantiate model, optimizer and hyperparameters
lr = 1e-4
num_epochs = 40

model = MNISTConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epochs in range(num_epochs):
    running_loss = 0.0
    num_correct = 0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        _, idx = outputs.max(dim=1)
        num_correct += (idx == labels).sum().item()
    print(
        "Loss: {}, Accuracy {}".format(
            running_loss / len(trainloader), num_correct / len(trainloader)
        )
    )

if __name__ == "__main__":
    True
