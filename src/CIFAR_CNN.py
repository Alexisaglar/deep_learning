from logging import warn
import matplotlib.pyplot as plt
import torch
from torch import feature_dropout, no_grad, optim
import torch.nn as nn
from torch.nn.modules import BatchNorm2d, Linear
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        x = self.block1(x)
        return self.block2(x)


# load in CIFAR dataset from PyTorch
traindataset = CIFAR10(".", train=True, download=True, transform=ToTensor())
testdataset = CIFAR10(".", train=False, download=True, transform=ToTensor())

trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
testloader = DataLoader(testdataset, batch_size=64, shuffle=False)


# instantiate model, optimizer and hyperparameters
lr = 1e-4
num_epochs = 400

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

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
        "Loss: {}, Accuracy {}, Epoch: {}".format(
            running_loss / len(trainloader), num_correct / len(trainloader), epochs
        )
    )

torch.save(model, "model.pth")
torch.save(model.state_dict(), "model_state_dict.pth")


if __name__ == "__main__":
    True
