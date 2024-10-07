import torch
import torch.nn as nn


class RelationClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 19)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):  # x (batch_size, input_dim)
        x = self.relu(self.fc1(x))  # (batch_size, 256)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  # (batch_size, 64)
        x = self.dropout(x)
        x = self.fc3(x)  # (batch_size, 20)
        return x  # (batch_size, 20)