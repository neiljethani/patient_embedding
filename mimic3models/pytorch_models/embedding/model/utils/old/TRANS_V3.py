import torch.nn as nn
import torch
import math, copy
import torch.nn.functional as F
from torch.autograd import Variable
from torch import tanh
import numpy as np



#### TRANSFORMER CLASS ####
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, embed=False):
        super(EncoderDecoder, self).__init__()
        self.encoder_A = encoder
        self.encoder_B = copy.deepcopy(encoder)
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.info = {'d_model': self.encoder_A.layers[0].size,
                     'layers': len(self.encoder_A.layers)}
        self.embed = embed
        
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        if not self.embed:
            z, x = self.encode(src, src_mask)
            z = self.decode(z, src_mask, tgt, tgt_mask)
            return x, z
        else:
            return self.embedding(src)
    
    def encode(self, src, src_mask):
        x = self.encoder_A(self.src_embed(src), src_mask)
        return self.encoder_B(x, src_mask), x 
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def embedding(self, src):
        return(self.encoder_B(self.encoder_A(self.src_embed(src), None), None))

#Purely an Encoder Model
class EncoderModel(nn.Module):
    def __init__(self, encoder, src_embed, generator, embed=False):
        super(EncoderModel, self).__init__()
        self.encoder_A = encoder
        self.encoder_B = copy.deepcopy(encoder)
        self.src_embed = src_embed
        self.generator = generator
        self.info = {'d_model': self.encoder_A.layers[0].size,
                     'layers': len(self.encoder_A.layers)}
        self.embed = embed
    
    def forward(self, src, src_mask=None):
        x, z = self.encode(src, src_mask)
        if not self.embed:
            return x, z
        else:
            return z
        
    def encode(self, src, src_mask):
        x = self.encoder_A(self.src_embed(src), src_mask)
        return x, self.encoder_B(x, src_mask) 
    
    def embedding(self, src):
        x, z = self.encode(src, None)
        return z
    
class EncoderModel_V2(nn.Module):
    def __init__(self, add_emb, add_connections, encoder, src_embed, generator, embed=False):
        super(EncoderModel_V2, self).__init__()
        self.add_emb = add_emb
        self.add_connections = add_connections
        self.encoder_A = encoder
        self.encoder_B = copy.deepcopy(encoder)
        self.src_embed = src_embed
        self.generator = generator
        self.info = {'d_model': self.encoder_A.layers[0].size,
                     'layers': len(self.encoder_A.layers)}
        self.embed = embed
    
    def forward(self, src, src_mask=None):
        src_mask = self.add_connections(src, src_mask)
        src = self.add_emb(src)
        x, z = self.encode(src, src_mask)
        if not self.embed:
            return x[:,:-1,:], z[:,:-1,:]
        else:
            return z[:,-1,:]
        
    def encode(self, src, src_mask):
        x = self.encoder_A(self.src_embed(src), src_mask)
        return x, self.encoder_B(x, src_mask) 
    
    
    
#### GENERAL UTILS ####
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()#GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Generator(nn.Module):
    "Define Standard Linear w/ Tanh instead of GELU activation"
    def __init__(self, d_model, d_input):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_input)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.proj(x))
      
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



#### EMBEDDING ####
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
    def __init__(self, d_model, device, max_len=25):
        super(PositionalEmbedding, self).__init__()
        
        self.lut = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        self.max_len = max_len
        self.device = device
        
    def forward(self, x):
        x = x + Variable(self.lut(torch.LongTensor(range(self.max_len)).to(self.device)))
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

#Mask and EMB Token Embeddings    
class InputTypeEmbedding(nn.Module):
    def __init__(self, d_input):
        super(InputTypeEmbedding, self).__init__()
        self.lut = nn.Embedding(3, d_input, padding_idx=0)
        self.d_input = d_input
        
    def forward(self, x):
        return self.lut(x)

#Adding all Embedding    
class CompleteEmbedding(nn.Module):
    def __init__(self, d_model, d_input):
        super(CompleteEmbedding, self).__init__()
        self.input = InputTypeEmbedding(d_input)
        self.linear = LinearEmbedding(d_model, d_input)
        self.positional = PositionalEmbedding(d_model)
        self.d_input = d_input
        self.d_model = d_model
    
    def forward(x, input_info=None, is_src=True):
        if is_src:
            x = x + self.input(input_info)
            x = self.linear(x)
            return self.positional(x)
        else:
            x = self.linear(x)
            return self.positional(x)


#### ATTENTION ####
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    

#### ENCODING ####
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
       
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    
    
#### DECODING ####
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
#### INPUT: Adding EMB Token ####
class InputAddToken(nn.Module):
    def __init__(self):
        super(InputAddToken, self).__init__()
        
    def forward(self, x, mask):
        #Pad Sequence with EMB Token
        x = F.pad(x, (0,0,0,1,0,0), 'constant', 0)
        #Pad Mask with 2 to create Seqment Info
        x_info = F.pad(mask, (0,1), 'constant', 2)
        mask = F.pad(mask, (0,1), 'constant', 0)
        return x, x_info, mask
    
class InputAddAverageToken(nn.Module):
    def __init__(self):
        super(InputAddAverageToken, self).__init__()
    
    def forward(self, x):
        ##Add Emb Token as Average to Sequence
        x = torch.cat((x, x.mean(1).unsqueeze(1)), dim=1)
        return x
    
    
#### Create Convolutional SRC Masks
class CreateConections(nn.Module):
    def __init__(self, device='cpu', seq_len=24):
        super(CreateConections, self).__init__()
        self.device = device
        self.seq_len=seq_len
    
    def forward(self, x, src_mask):
        if src_mask == None:
            src_mask = np.ones((x.size()[0], self.seq_len, self.seq_len))
        else:
            src_mask.numpy()
        #Add Connections to Neighbors
        src_mask = np.triu(src_mask, -1)
        src_mask[np.triu(src_mask, 2)==1] = 0
        src_mask = torch.tensor(src_mask, device=self.device).float()
        
        #Append Connections to Embedding Token
        src_mask = F.pad(src_mask, (0,1,0,1,0,0), 'constant', 1)
        
        return src_mask
        
            
    


