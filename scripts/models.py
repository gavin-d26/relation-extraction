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
    

class Block(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(in_features=in_features, 
                                            out_features=out_features,
                                            bias=True),
                                #   nn.BatchNorm1d(out_features),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))
        
    def forward(self, inputs):
        return self.layers(inputs)

class RelationClassifierPro(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(Block(input_dim, 256, dropout=dropout),
                                    Block(256, 64, dropout=dropout),
                                    Block(64, 64, dropout=dropout),
                                    nn.Linear(64, 19)
                                    )
    
    def forward(self, inputs):  # x (batch_size, input_dim)
        return self.layers(inputs)