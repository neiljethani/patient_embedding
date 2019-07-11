import torch
import torch.nn as nn
import copy

import pickle

from .classifier import *

def make_model(d_input = None, embed_model = None, max_len = 24, dropout=0.1, embed_method='TRANS'):
        
    if embed_method == 'TRANS':
        
        embed_model = torch.load(embed_model)
        embed_model.eval()
        
        for p in embed_model.parameters():
            p.requires_grad = False
        embed_model.embed = True
        
        d_model = embed_model.info['d_model']
        model = nn.Sequential(
            embed_model,
            TransformerClassifier(d_model=d_model, seq_len=max_len, dropout=dropout)
        )
                
    elif embed_method == 'DAE':
        
        embed_model = torch.load(embed_model)
        embed_model.eval()
        
        for p in embed_model.parameters():
            p.requires_grad = False
        embed_model.embed = True
        
        d_model = embed_model.info['d_model']
        model = nn.Sequential(
            embed_model,
            DAEClassifier(d_model=d_model, dropout=dropout)
        )
        
    elif embed_method == 'PCA':
        
        embed_model = pickle.load(open(embed_model, 'rb'))
        
        d_model = embed_model.info['d_model']
        model = PCAClassifier(PCAmodel = embed_model, d_model=d_model, dropout=dropout)
                
    elif embed_method == 'RAW':
        
        d_model = int(d_input*max_len/4)
        d_input = d_input*max_len
        model = RawClassifier(d_input, d_model, dropout=0.1)
        
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)
    
    return model, d_model