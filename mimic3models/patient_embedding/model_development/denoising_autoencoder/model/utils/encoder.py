import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, seq_len, deepN = None):
        super(Encoder, self).__init__()
        self.encode = nn.Linear(d_input*seq_len, d_model)
        if deepN is not None:
            self.linears = nn.ModuleList([nn.Linear(d_model*seq_len, d_model*seq_len) for _ in range(deepN)])
            self.deep = True
        else:
            self.deep = False
        
    def forward(self, x):
        x = F.relu(self.encode(x))
        if self.deep:
            for layer in self.linears:
                x = F.relu(layer(x))
        return x  