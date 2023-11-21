import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularModel(nn.Module):
    def __init__(self, num_features):
        super(TabularModel, self).__init__()
        self.layer1 = nn.Linear(num_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.layer3(x))

        return torch.sigmoid(self.output(x))
