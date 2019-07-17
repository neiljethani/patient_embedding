import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model.model import make_model
from .utils import NoamOpt
from .utils import MedEmbedLoss

import os
import shutil
import arrow
import pickle

import tqdm



class EmbeddingTrainer():
    """
    General Trainer for all Model Types
    """
    def __init__(self, output_dir, train_dataloader: DataLoader, test_dataloader: DataLoader = None, 
                 embed_method: str = 'TRANS', model = None, 
                 layers: int = 3, heads: int = 4, dropout: float = 0.1, MSprop: float = 0.5,
                 lr: float = 0, betas = (0.9, 0.98), eps=1e-9, 
                 factor: int = 2, warmup: int = 4000, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        
        """
        :param embed_method: Model Type to Use for Embedding
        :param model: Model to use, otherwise make_model is used
        :param layers: Number of Layers to use for Encoding/Decoding
        :param Dropout: Dropout/Noise to use
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param eps: Adam optimizer eps
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        
        #Set Embedding Method
        self.embed_method = embed_method 
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print(self.device)
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        #Initialize Model
        if model is not None:
            self.model = model.float()
        else:
            self.model = make_model(d_input = self.train_data.dataset.get_input_dim(), 
                                    N=layers, max_len= self.train_data.dataset.get_seq_length(), 
                                    dropout=dropout, embed_method = embed_method)
            if embed_method != 'PCA':
                self.model = self.model.float()
                self.model.to(self.device)
                print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        if embed_method != 'PCA':
            #Set Optimizer and Scheduler
            self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
            self.optim_schedule = NoamOpt(model_size = self.model.info['d_model'], 
                                          factor = factor, warmup = warmup, optimizer = self.optim)
            
            #Set Loss criterion + Function + Proportion of Masked Sequence Prediction in Loss
            self.criterion = nn.MSELoss()
            if self.embed_method == 'TRANS':
                self.loss_fn = MedEmbedLoss(criterion=self.criterion, generator = self.model.generator, 
                                            opt=self.optim_schedule, embed_method = embed_method)
                self.MSprop = MSprop
            else:
                self.loss_fn = MedEmbedLoss(criterion=self.criterion, generator = None, 
                                            opt=self.optim_schedule, embed_method = embed_method)
        else:
            self.criterion = nn.MSELoss()
                
        #Create save dir + best dir
        self.save_dir = os.path.join(output_dir, self.embed_method, 
                                     arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) 
            
        self.best_dir = os.path.join(self.save_dir, "best")
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
               
        #Initialize Loses
        self.train_loss = {}
        self.test_loss = {}
        
        #Set Log Frequency
        self.log_freq = log_freq
        
    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        total_loss = 0
        for i, data in data_iter:
            #Send Batched Data to device (cpu or gpu)
            data = {key: value.float().to(self.device) for key, value in data.items()}
            
            #Run Data Through the Model
            if self.embed_method == 'DAE':
                tgt_pred = self.model(src=data['src'])

                #Calculate Loss and Take a Step
                loss = self.loss_fn(data['tgt'], tgt_pred, data['norm'], train=train)

                #Track Total Loss
                total_loss += loss
                
            elif self.embed_method == 'TRANS':
                src_pred, tgt_pred = self.model(src=data['src_masked'], tgt=data['tgt_input'], 
                                                src_mask=None, tgt_mask=data['tgt_mask'])
            
                #Calculate Loss and Take a Step
                loss = self.loss_fn(MSx=src_pred, MSy=data['src'], MSmask=data['mask'], 
                                    FSx=tgt_pred, FSy=data['tgt'], 
                                    norm=data['norm'], MSprop=self.MSprop, train=train)

                #Track Total Loss
                total_loss += loss
            
            elif self.embed_method == 'PCA':
                #Run Data Through the Model
                if train:
                    self.model.partial_fit(data['src'].numpy())
                tgt_pred = self.model.inverse_transform(self.model.transform(data['src']))

                #Calculate Loss Loss
                loss = self.criterion(torch.tensor(tgt_pred).float().contiguous(), 
                                      data['tgt'].contiguous()).item()
                
                #Track Total Loss
                total_loss += loss
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "total_loss": total_loss,
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        
        if self.embed_method == 'PCA':
            avg_loss = total_loss/len(data_iter)
        else:
            avg_loss = total_loss/data_loader.dataset.get_number_of_visits()
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss)
        
        if train:
            self.train_loss[epoch] = avg_loss
        else:
            self.test_loss[epoch] = avg_loss
        
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
        
    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = os.path.join(self.save_dir, "{}.ep{}".format(self.embed_method, epoch))
        if self.embed_method == 'PCA':
            pickle.dump(self.model, open(output_path, 'wb'))
        else:
            torch.save(self.model.cpu(), output_path)
            self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def save_best(self):
        """
        After all Models are saved store best performing epoch
        
        :param epoch: best epoch number
        :return: final_output_path
        """
        epoch = min(self.test_loss, key=self.test_loss.get)
        src_path = os.path.join(self.save_dir, "{}.ep{}".format(self.embed_method, epoch))
        best_path = os.path.join(self.best_dir, "{}.ep{}".format(self.embed_method, epoch))
        
        #Copy Model to Best Directory
        shutil.copyfile(src_path, best_path)
        
    def write_loss(self):
        """
        save training and test losses for each epoch as csv.
        
        :return: lossfile.csv
        """
        
        with open(os.path.join(self.save_dir, 'lossfile.csv'), 'w') as lossfile:
            lossfile.write("epoch, train_loss, test_loss\n")
            i = 0
            while i in self.train_loss:
                lossfile.write('{},{},{}\n'.format(str(i), str(self.train_loss[i]), str(self.test_loss[i])))
                i += 1

            
                

            
        
        