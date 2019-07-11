import numpy as np
import torch
import torch.nn as nn



#### LOSS ####
class LossCompute:
    "A loss compute and normalized by window size."
    def __init__(self, criterion, generator = None, embed_method = 'TRANS'):
        self.embed_method = embed_method
        self.generator = generator
        self.criterion = criterion
        
    def __call__(self, x, y, norm = None, mask = None):
        loss = 0
        
        if self.embed_method == 'TRANS':
            x = self.generator(x)
        else:
            mask = None
        
        if mask is not None:
            if norm is not None:
                for i, j, m, n in zip(x, y, mask, norm):
                    i = i[(m==1)]
                    j = j[(m==1)]
                    loss += self.criterion(i.contiguous(), j.contiguous())/n
            else:
                for i, j, m, n in zip(x, y, mask):
                    i = i[(m==1)]
                    j = j[(m==1)]
                    loss += self.criterion(i.contiguous(), j.contiguous())
        else:
            if norm is not None:
                for i, j , n in zip(x, y, norm):
                    loss += self.criterion(i.contiguous(), j.contiguous())/n
            else:
                for i, j in zip(x, y):
                    loss += self.criterion(i.contiguous(), j.contiguous())
            
        return loss


class MedEmbedLoss:
    "Class to Calculate Embedding Loss: Masked Sequence Loss + Future Sequence Loss"
    def __init__(self, criterion, generator = None, opt=None, embed_method = 'TRANS'):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.loss = LossCompute(criterion, generator, embed_method)
        self.embed_method = embed_method
        
    def __call__(self, MSx, MSy, norm, MSmask = None, FSx = None, FSy = None, MSprop=0.5, train=True):
        if self.embed_method == 'TRANS':
            MSLoss = self.loss(MSx, MSy, norm, MSmask)
            FSLoss = self.loss(FSx, FSy, norm)
            loss = MSprop*MSLoss + (1-MSprop)*FSLoss
        else:
            loss = self.loss(MSx, MSy, norm)
        
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        
        return loss.item()
    

    
    
#### OPTIMIZER SCHEDULE ####
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
