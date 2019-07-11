import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


class TransformerClassifier(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.conv = nn.Conv1d(seq_len, seq_len, d_model)
        self.hidden = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_weights = F.softmax(F.relu(self.conv(x)), dim=1).transpose(1,2)
        x = torch.bmm(attn_weights, x)
        x = F.relu(self.dropout(self.hidden(x)))
        x = F.softmax(self.output(x)) 
        return x
      
    
class DAEClassifier(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DAEClassifier, self).__init__()
        self.hidden = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.dropout(self.hidden(x)))
        x = F.softmax(self.output(x))
        return x

    
class PCAClassifier(DAEClassifier):
    def __init__(self, PCAmodel, d_model, dropout=0.1):
        super(PCAClassifier, self).__init__(d_model=d_model, dropout=dropout)
        self.PCAmodel = PCAmodel
    
    def forward(self, x):
        x = self.PCAmodel.embedding(x)
        x = F.relu(self.dropout(self.hidden(x)))
        x = F.softmax(self.output(x))
        return x
    
        
class RawClassifier(nn.Module):
    def __init__(self, d_input, d_model, dropout=0.1):
        super(RawClassifier, self).__init__()
        self.embed = nn.Linear(d_input, d_model)
        self.hidden = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, 2)
        self.embed_dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.embed_dropout(self.embed(x)))
        x = F.relu(self.hidden_dropout(self.hidden(x)))
        x = F.softmax(self.output(x))
        return x    