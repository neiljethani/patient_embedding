import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, noise = None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise = noise
        self.info = {'d_model': self.encoder.encode.out_features, 
                     'noise': noise}
    
    def forward(self, src):
        return self.decode(self.encode(src))
    
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
        