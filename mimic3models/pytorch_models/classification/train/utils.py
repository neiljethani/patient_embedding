import numpy as np
import torch
import torch.nn as nn


#### LOSS ####
class LossCompute:
    "A loss compute and normalized by window size."
    def __init__(self, criterion, opt):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y,  train = True):
        x = x.squeeze(1)[:, 1]
        loss = self.criterion(x.contiguous(), y.contiguous())
        
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.zero_grad()
                #self.opt.optimizer.zero_grad()
            
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
