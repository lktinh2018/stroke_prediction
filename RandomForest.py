import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionTree(nn.Module):
    def __init__(self, n_features):
        super(DecisionTree, self).__init__()
        self.layer1 = nn.Linear(n_features, 10)
        self.layer2 = nn.Linear(10, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


# Random Forest Layers
class RandomForest(nn.Module):
    def __init__(self, num_trees=10):
        super(RandomForest, self).__init__()
        self.trees = nn.ModuleList([DecisionTree(num_trees) for _ in range(num_trees)])

    def forward(self, x):
        # Combine prediction
        tree_outputs = [tree(x) for tree in self.trees]

        return torch.stack(tree_outputs).mean(0)


# # Init model
# random_forest = RandomForest(num_trees=10)

# # Sample Data
# x = torch.randn(5, 3)  # Giả sử bạn có 5 mẫu dữ liệu với 3 đặc trưng

# # Prediction
# predictions = random_forest(x)
