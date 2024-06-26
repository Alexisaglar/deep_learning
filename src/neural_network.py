import torch.nn as nn
import torch
from torch import optim


class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
        self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        out = self.layer2(x)
        return out


no_examples = 10
in_dim, feature_dim, out_dim = 784, 256, 10
x = torch.randn((no_examples, in_dim))
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
out = classifier(x)

loss = nn.CrossEntropyLoss()
target = torch.tensor([0, 3, 2, 8, 2, 9, 3, 7, 1, 6])
computed_loss = loss(out, target)
computed_loss.backward()

for p in classifier.parameters():
    print(p.shape)

lr = 1e-3
optimizer = optim.SGD(classifier.parameters(), lr=lr)
optimizer.step()
optimizer.zero_grad()
