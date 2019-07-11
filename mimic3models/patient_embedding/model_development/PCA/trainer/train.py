import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.model import make_model


import os
import arrow
import tqdm

class PCATrainer():
    def __init__(self, train_dataloader: DataLoader, model = None, 
                 test_dataloader: DataLoader = None, 
                 log_freq: int = 10):
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        #Initialize Model
        if model is not None:
            self.model = model.float()
        print(self.train_data.dataset.get_input_dim())
        print(self.train_data.dataset.get_seq_length())
        self.model = make_model(d_input = self.train_data.dataset.get_input_dim(), 
                                max_len= self.train_data.dataset.get_seq_length()).float()
        
        self.criterion = nn.MSELoss()
        
        #Set Log Frequency
        self.log_freq = log_freq
        
        #Create save dir
        self.save_dir = os.path.join('/work/MIMIC/models/patient_embedding/PCA', 
                                     arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        
    def train(self):
        
        str_code = "train"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        for i, data in data_iter:
            #Send Batched Data to device (cpu or gpu)
            data = {key: value.float().to(self.device) for key, value in data.items()}
            
            #Run Data Through the Model
            self.model.partial_fit(data['src'].numpy())
            tgt_pred = self.model.inverse_transform(self.model.transform(data['src']))
            
            total_loss += self.criterion(torch.tensor(tgt_pred).float().contiguous(), 
                                         data['tgt'].contiguous())
            
            post_fix = {
                "iter": i,
                "total_loss": total_loss,
                "loss": loss
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                
        print("EP%d_%s, avg_loss=" % (epoch, str_code), total_loss/len(self.train_data))
        
    def test(self):
        
        str_code = "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        for i, data in data_iter:
            #Send Batched Data to device (cpu or gpu)
            data = {key: value.float().to(self.device) for key, value in data.items()}
            
            #Run Data Through the Model
            tgt_pred = self.model.inverse_transform(self.model.transform(data['src']))
            
            total_loss += self.criterion(torch.tensor(tgt_pred).float().contiguous(), 
                                         data['tgt'].contiguous())
            
            post_fix = {
                "iter": i,
                "total_loss": total_loss,
                "loss": loss
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                
        print("EP%d_%s, avg_loss=" % (epoch, str_code), total_loss/len(self.train_data))
            
    def save(self, epoch):
        """
        Saving the current model on file_path
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = self.save_dir + ".skm"
        pickle.dump(self.model, open(output_path, 'wb'))
        print("Model Saved on:" output_path)
        return output_path
    
    
            
            