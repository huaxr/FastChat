import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

