import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh
import math, copy, time
from torch.autograd import Variable



#### DAE CLASS ####
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, noise = None, embed = False):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise
        self.info = {'d_model': self.encoder.encode.out_features, 
                     'noise': noise}
        self.embed = embed
    
    def forward(self, src):
        if not self.embed:
            return self.decode(self.encode(src))
        else: 
            return self.embedding(src)
    
    def encode(self, src):
        if self.noise is not None:
            return self.encoder(self.noise(src))
        else:
            return self.encoder(src)
    
    def decode(self, memory):
        return self.decoder(memory)
    
    def embedding(self, src):
        return self.encoder(src)
    
    def info(self):
        return self.info



#### ENCODING ####
class Encoder(nn.Module):
    def __init__(self, d_input, d_model, seq_len, deepN = None):
        super(Encoder, self).__init__()
        self.encode = nn.Linear(d_input*seq_len, d_model)
        if deepN is not None:
            self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(deepN)])
            self.deep = True
        else:
            self.deep = False
        
    def forward(self, x):
        x = tanh(self.encode(x))
        if self.deep:
            for layer in self.linears:
                x = tanh(layer(x))
        return x  
    
    
    
#### DECODING ####
class Decoder(nn.Module):
    def __init__(self, d_input, d_model, seq_len, deepN = None):
        super(Decoder, self).__init__()
        self.decode = nn.Linear(d_model, d_input*seq_len)
        if deepN is not None:
            self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(deepN)])
            self.deep = True
        else:
            self.deep = False
    
    def forward(self, x):
        if self.deep:
            for layer in self.linears:
                x = tanh(layer(x))
        x = tanh(self.decode(x))
        return x
    

    
