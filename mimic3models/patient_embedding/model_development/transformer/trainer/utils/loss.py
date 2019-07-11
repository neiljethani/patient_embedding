import numpy as np
import torch
import torch.nn as nn

class LossCompute:
    "A loss compute and normalized by window size."
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion
        
    def __call__(self, x, y, norm, mask = None):
        x = self.generator(x)
        loss = 0
        if mask is not None:
            for i, j, m, n in zip(x, y, mask, norm):
                i = i[(m==1)]
                j = j[(m==1)]
                loss += self.criterion(i.contiguous(), j.contiguous()) / n
        else:
            for i, j, n in zip(x, y, norm): 
                loss += self.criterion(i.contiguous(), j.contiguous()) / n
        
        return loss


class MedEmbedLoss:
    "Class to Calculate Embedding Loss: Masked Sequence Loss + Future Sequence Loss"
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.loss = LossCompute(generator, criterion)
        
    def __call__(self, MSx, MSy, MSmask, FSx, FSy, norm, MSprop=0.5, train=True):
        MSLoss = self.loss(MSx, MSy, norm, MSmask)
        FSLoss = self.loss(FSx, FSy, norm)
        loss = MSprop*MSLoss + (1-MSprop)*FSLoss
        
        if train:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
        
        return loss.item()