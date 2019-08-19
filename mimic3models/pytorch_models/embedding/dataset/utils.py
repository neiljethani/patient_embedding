import torch
from torch.utils import data

import numpy as np
import copy

from ..model.utils.TRANS import subsequent_mask

#Pytorch Wrapper for load_data --> Feeds of PatientEmbeddingReader
class PatientEmbeddingDataset(data.Dataset):
    def __init__(self, reader, discretizer, embed_method = 'TRANS', normalizer=None, mask_percent=.15, return_name=False):
        self.reader = reader
        self.discretizer = discretizer
        self.normalizer = normalizer
        self.mask_percent = mask_percent
        self.return_name = return_name
        self.embed_method = embed_method
        self.n_visits = self.reader.get_number_of_visits()
        self.input_dim = self.reader.get_input_dim()
        self.seq_length = int(self.reader._period_length/2)
    
    def __len__(self):
        return self.reader.get_number_of_examples()
    
    def get_number_of_visits(self):
        return self.n_visits
    
    def get_input_dim(self):
        return self.input_dim
    
    def get_seq_length(self):
        return self.seq_length
    
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
        tgt = np.array(X[-int(t/2):])
        
        if self.embed_method in ['TRANS', 'COPY']:
            tgt = np.array(X[-int(t/2):])
            
            if self.embed_method == 'COPY':
                data = {'src':src, 'tgt':tgt, 'norm':norm}
                if self.return_name:
                    data['name'] = name
                return data

            mask = np.zeros(int(t/2))
            if self.mask_percent > 0.01:
                n_masks = round(self.mask_percent*int(t/2))
                mask_ids = np.random.permutation(int(t/2))[:n_masks]
                mask[mask_ids] = 1

            src_masked = copy.deepcopy(src)
            src_masked[(mask==1), :] = 0

            tgt_input = np.vstack((src[-1,:], tgt[:-1,:]))
            tgt_mask = subsequent_mask(tgt_input.shape[0]).squeeze(0)
            
            data = {'src_masked':src_masked, 'src':src, 
                    'tgt_input':tgt_input, 'tgt':tgt, 'tgt_mask': tgt_mask,
                    'norm':norm, 'mask':mask} 
        elif self.embed_method in ['PCA', 'DAE']:
            src = src.flatten()
            data = {'src':src, 'tgt':src, 'norm':norm}
        elif self.embed_method == 'DFE':
            src = src.flatten()
            tgt = tgt.flatten()
            data = {'src':src, 'tgt':tgt, 'norm':norm}
        
        if self.return_name:
            data['name'] = name
        
        return data 
    

        