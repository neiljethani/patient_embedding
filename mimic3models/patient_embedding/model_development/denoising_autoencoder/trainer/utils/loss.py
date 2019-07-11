mport numpy as np
import torch
import torch.nn as nn

class LossCompute:
    def __init__(self, criterion, opt = None):
        self.criterion = criterion
        self.opt = opt 
        
    def __call__(self, x, y, norm = None, train = True):
        loss = 0
        if norm is not None:
            for i, j , n in zip(x, y, norm):
                loss += self.criterion(i.contiguous(), j.contiguous())/n
        else:
            for i, j in zip(x, y):
                loss += self.criterion(i.contiguous(), j.contiguous())

        
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
                
        return loss.item()
            