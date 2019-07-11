import torch
from sklearn.decomposition import IncrementalPCA

class PCA(IncrementalPCA):
    def __init__(self, n_components=None, whiten=False, copy=True, batch_size=None):
        super(PCA, self).__init__(n_components=n_components, whiten=whiten, copy=copy, batch_size=batch_size)
        self.info = {'d_model': n_components}
    def embedding(self, X):
        return(torch.Tensor(self.transform(X)).float())
    def info(self)
        return self.info
              
def make_model(d_input, max_len = 24):
    #Define Dimensions
    d_model = int(d_input*max_len/4)
    
    model = PCA(n_components=d_model)
    
    return model