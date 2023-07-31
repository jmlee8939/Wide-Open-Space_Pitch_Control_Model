import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import optim

class nnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4, 24, bias=True)
        self.hidden2 = nn.Linear(24, 16, bias= True)
        self.hidden3 = nn.Linear(16, 8, bias= True)
        self.output = nn.Linear(8, 1, bias=True)
        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.output(x))

        return x
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.hidden1.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.hidden2.weight)
        self.hidden2.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.hidden3.weight)
        self.hidden3.bias.data.fill_(0.01)
        nn.init.kaiming_normal_(self.output.weight)
        self.output.bias.data.fill_(0.01)