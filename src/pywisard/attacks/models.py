import torch.nn as nn


class SurrogateMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size2, num_classes),
        )

    def forward(self, x):
        return self.network(x)
