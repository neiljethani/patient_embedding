import torch
import torch.nn as nn
import copy

from .utils.attention import MultiHeadedAttention
from .utils.embedding import LinearEmbedding, PositionalEmbedding
from .utils.general_utils import PositionwiseFeedForward, Generator
from .utils.transformer import EncoderDecoder
from .utils.encoder import Encoder, EncoderLayer
from .utils.decoder import Decoder, DecoderLayer

def make_model(d_input, N=3, h=4, max_len = 24, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    #Define Dimensions
    d_model = d_input*h
    d_ff = d_model*4
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEmbedding(d_model, max_len)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N*2),
        nn.Sequential(LinearEmbedding(d_model, d_input), c(position)),
        nn.Sequential(LinearEmbedding(d_model, d_input), c(position)),
        Generator(d_model, d_input))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model