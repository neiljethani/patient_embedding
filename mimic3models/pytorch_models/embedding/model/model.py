import torch
import torch.nn as nn
import copy

from .utils import TRANS
from .utils import DAE
from .utils.PCA import PCA



def make_model(d_input, N=3, h=6, max_len = 24, dropout=0.1, embed_method='TRANS', device = 'cpu'):
    if embed_method == 'TRANS':
        print('making model')
        d_model = d_input*h
        d_ff = d_input

        c = copy.deepcopy
        attn = TRANS.MultiHeadedAttention(h, d_model)
        ff = TRANS.PositionwiseFeedForward(d_model, d_ff, dropout)
        position = TRANS.PositionalEmbedding(d_model, device, max_len+2)

        model = TRANS.EncoderModel(
            TRANS.InputAddAverageToken(),
            TRANS.CreateConections(seq_len=max_len, device=device),
            TRANS.Encoder(TRANS.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            nn.Sequential(TRANS.LinearEmbedding(d_model, d_input), c(position)),
            TRANS.Generator(d_model, d_input*max_len))
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        model = model.float().to(device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
                
    elif embed_method == 'DAE':
        d_model = int(d_input*max_len/4)
        
        if N == 0:
            N = None
    
        Noise = nn.Dropout(dropout) if dropout is not None else None
        model = DAE.EncoderDecoder(
            DAE.Encoder(d_input, d_model, max_len, N),
            DAE.Decoder(d_input, d_model, max_len, N),
            Noise
        )
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        model = model.float().to(device)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
                
    elif embed_method == 'PCA':
        #Define Dimensions
        d_model = int(d_input*max_len/4)
        model = PCA(n_components=d_model)
        
    else:
        model = None
    
    return model