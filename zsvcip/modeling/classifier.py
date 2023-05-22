import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_features=512, hidden=512, dropout=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x)
