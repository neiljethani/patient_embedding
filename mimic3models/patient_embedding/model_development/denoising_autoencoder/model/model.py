import torch
import torch.nn as nn
import copy

from .utils.denoising_autoencoder import EncoderDecoder
from .utils.encoder import Encoder
from .utils.decoder import Decoder

def make_model(d_input, N = None, max_len = 24, dropout=0.15):
    #Define Dimensions
    d_model = int(d_input*max_len/4)
    
    Noise = nn.Dropout(dropout) if dropout is not None else None
    model = EncoderDecoder(
        Encoder(d_input, d_model, max_len, N),
        Decoder(d_input, d_model, max_len, N),
        Noise
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model