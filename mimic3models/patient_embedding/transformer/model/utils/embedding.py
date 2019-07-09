import torch.nn as nn
import torch
from torch.autograd import Variable
import math



#Position Embeddings
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class PositionalEmbedding(nn.Module):
    "Postional Embedding Class, where Positional Embedding is Learned"
    def __init__(self, d_model, max_len=24):
        super(PositionalEmbedding, self).__init__()
        
        self.lut = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        x = x + Variable(self.lut(torch.LongTensor(range(self.max_len))))
        return x

#Input Embeddings    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class LinearEmbedding(nn.Module):
    def __init__(self, d_model, d_input):
        super(LinearEmbedding, self).__init__()
        self.lut = nn.Linear(d_input, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
