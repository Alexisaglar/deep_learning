import os
from typing_extensions import NamedTuple
import torch
from torch.utils import data
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib

# dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
# download_url(dataset_url, ".")
# with tarfile.open("./cifar10.tgz", "r:gz") as tar:
#     tar.extractall(path="./data")

# read data from directory


def show_example(img, label):
    matplotlib.rcParams["figure.facecolor"] = "#ffffff"
    print("label: ", dataset.classes[label], "(" + str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    return


data_dir = "data/cifar10"
classes = os.listdir(data_dir + "/train")
dataset = ImageFolder(data_dir + "/train", transform=ToTensor())

# delimiting dataset for training and validation data
random_seed = 42
torch.manual_seed(random_seed)
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
print(len(train_ds), len(val_ds))


# if __name__ == "__main__":
#     True
#     # show_example(*dataset[0])
