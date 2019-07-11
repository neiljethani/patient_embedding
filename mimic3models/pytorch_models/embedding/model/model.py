import torch
import torch.nn as nn
import copy

from .utils import TRANS
from .utils import DAE
from .utils.PCA import PCA



def make_model(d_input, N=3, h=4, max_len = 24, dropout=0.1, embed_method='TRANS'):
    if embed_method == 'TRANS':
        print('making model')
        d_model = d_input*h
        d_ff = d_model*4

        c = copy.deepcopy
        attn = TRANS.MultiHeadedAttention(h, d_model)
        ff = TRANS.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = TRANS.PositionalEmbedding(d_model, max_len)
        model = TRANS.EncoderDecoder(
            TRANS.Encoder(TRANS.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            TRANS.Decoder(TRANS.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N*2),
            nn.Sequential(TRANS.LinearEmbedding(d_model, d_input), c(position)),
            nn.Sequential(TRANS.LinearEmbedding(d_model, d_input), c(position)),
            TRANS.Generator(d_model, d_input))
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    elif embed_method == 'DAE':
        d_model = int(d_input*max_len/4)
    
        Noise = nn.Dropout(dropout) if dropout is not None else None
        model = DAE.EncoderDecoder(
            DAE.Encoder(d_input, d_model, max_len, N),
            DAE.Decoder(d_input, d_model, max_len, N),
            Noise
        )
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    elif embed_method == 'PCA':
        #Define Dimensions
        d_model = int(d_input*max_len/4)
        model = PCA(n_components=d_model)
    
    return model