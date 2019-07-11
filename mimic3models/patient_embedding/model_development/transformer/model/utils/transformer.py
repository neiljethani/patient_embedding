import torch.nn as nn
import copy

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder_A = encoder
        self.encoder_B = copy.deepcopy(encoder)
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.info = {'d_model': self.encoder_A.layers[0].size.
                     'layers': len(self.encoder_A.layers)}
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        z, x = self.encode(src, src_mask)
        z = self.decode(z, src_mask, tgt, tgt_mask)
        #x = self.generator(x)
        #z = self.generator(z)
        return x, z
    
    def encode(self, src, src_mask):
        x = self.encoder_A(self.src_embed(src), src_mask)
        return self.encoder_B(x, src_mask), x 
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def embedding(self, src):
        return(self.encoder_B(self.encoder_A(self.src_embed(src), None), None))
