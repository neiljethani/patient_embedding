import torch
from torch.utils import data

import numpy as np
import copy

#Pytorch Wrapper for load_data --> Feeds of InHospitalMortalityReader
#Embedding Type Taken as Input
class ClassificationDataset(data.Dataset):
    def __init__(self, embed_method, reader, discretizer, normalizer=None, return_name=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.return_name = return_name
        self.embed_method = embed_method
    
    def __len__(self):
        return self.reader.get_number_of_examples()
    
    def get_input_dim(self):
        return self.reader.get_input_dim()
    
    def get_seq_length(self):
        return int(self.reader._period_length/2)
    
    def __getitem__(self, index):
        ret = self.reader.read_example(index)
        
        X = ret["X"]
        t = ret["t"]
        y = ret["y"]
        name = ret["name"]
        
        X = self.discretizer.transform(X, end=24)[0][-int(t):]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        if self.embed_method != 'TRANS':
            X = X.flatten()
        
        if not self.return_name:
            return {'X':X, 'y':y}
        return {'X':X, 'y':y, 'name':name}