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
        
    def __call__(self, x, y, norm = None):
        loss = 0
        
        if self.embed_method == 'TRANS':
            x = self.generator(x)
           
        if norm is not None:
            for i, j, n in zip(x, y, norm):
                loss += self.criterion(i.contiguous(), j.contiguous())/n
        else:
            for i, j in zip(x, y):
                loss += self.criterion(i.contiguous(), j.contiguous())
            
        return loss


class MedEmbedLoss:
    "Class to Calculate Embedding Loss: Masked Sequence Loss + Future Sequence Loss"
    def __init__(self, criterion, model = None, opt=None, embed_method = 'TRANS'):
        if embed_method == 'TRANS':
            self.generator = model.generator
        else: 
            self.generator = None
        self.criterion = criterion
        self.opt = opt
        self.loss = LossCompute(self.criterion, self.generator, embed_method)
        self.embed_method = embed_method
        
    def __call__(self, MSx, MSy, norm = None, MSmask = None, FSx = None, FSy = None, MSprop=0.5, train=True):
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
                self.opt.zero_grad()
                
        
        if self.embed_method == 'TRANS':
            return loss.item(), MSLoss.item(), FSLoss.item()
        
        return loss.item()