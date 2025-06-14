import torch
import torch.nn as nn

class FakeNewsModel(nn.Module):
    def __init__(self, input_size=5000, hidden_size=128, output_size=2):
        super(FakeNewsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
