import torch
from torch.utils import data

import numpy as np
import copy


#Pytorch Wrapper for load_data --> Feeds of PatientEmbeddingReader
class PatientEmbeddingDataset(data.Dataset):
    def __init__(self, reader, discretizer, normalizer=None, return_name=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.return_name = return_name
    
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
        end = ret["end_time"]
        name = ret["name"]
        norm = np.array(ret["norm"])
        
        X = self.discretizer.transform(X, end=end)[0][-int(t):]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        src = np.array(X[-int(t):-int(t/2)])
        
        if not self.return_name:
            return {'src':src, 'tgt':src}
        return {'src':src, 'tgt':src, 'name':name}