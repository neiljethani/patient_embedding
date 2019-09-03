import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model.model import make_model
from .utils import NoamOpt
from .utils import MedEmbedLoss, LossCompute

import os
import shutil
import arrow
import pickle
import random

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
        :param heads: Number of attention heads to use (for TRANS only)
        :param MSprop: Proportion of Reconstruction Loss in total Loss (for TRANS only)
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
        cuda_condition = torch.cuda.is_available() and with_cuda and (embed_method != "PCA")
        self.device = torch.device(("cuda:0" if cuda_devices == None else cuda_devices)[0] if cuda_condition else "cpu")
        print(self.device)
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        #Set Reconstruction Loss Proportion (Only used if embed_method is TRANS)
        self.MSprop = MSprop
        
        #Initialize Model
        if model is not None:
            self.model = torch.load(model).float()
            if embed_method != 'PCA':
                self.model.to(self.device)
        else:
            self.model = make_model(d_input = self.train_data.dataset.get_input_dim(), h=heads,
                                    N=layers, max_len= self.train_data.dataset.get_seq_length(), 
                                    dropout=dropout, embed_method = embed_method, device = self.device)
        
        #Set Optimizer
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps) 
        
        #Set Criterion 
        self.criterion = nn.MSELoss()
        
        #Set Loss Function
        if embed_method in ['TRANS', 'DAE', 'DFE']:    
            self.loss_fn = MedEmbedLoss(criterion=self.criterion, model = self.model, 
                                        opt=self.optim, embed_method = embed_method)
            
            
        else:
            self.loss_fn = LossCompute(criterion=self.criterion, 
                                       generator = None, embed_method = embed_method)
            
        #Create save dir + best dir
        self.save_dir = os.path.join(output_dir, self.embed_method, 
                                     arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            self.save_dir = os.path.join(output_dir, self.embed_method, 
                                     arrow.now().format('YYYY-MM-DD_HH-mm-ss'))
            os.makedirs(self.save_dir)
            
        self.best_dir = os.path.join(self.save_dir, "best")
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
               
        #Initialize Loses
        self.train_loss = {}
        self.test_loss = {}
        self.train_MS = {}
        self.train_FS = {}
        self.test_MS = {}
        self.test_FS = {}
        
        #Set Log Frequency
        self.log_freq = log_freq
        
        #Create Dict of Lists to Store Mini-batches
        self.dataset_list = {'train':[], 'test':[]}
        
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
          
        if epoch == 0:
            # Setting the tqdm progress bar
            data_iter = tqdm.tqdm(enumerate(data_loader),
                                  desc="EP_%s:%d" % (str_code, epoch),
                                  total=len(data_loader),
                                  bar_format="{l_bar}{r_bar}")
        else:
            if train:
                random.shuffle(self.dataset_list['train'])
                data_iter = tqdm.tqdm(enumerate(self.dataset_list['train']),
                                      desc="EP_%s:%d" % (str_code, epoch),
                                      total=len(data_loader),
                                      bar_format="{l_bar}{r_bar}")
            else:
                data_iter = tqdm.tqdm(enumerate(self.dataset_list['test']),
                                      desc="EP_%s:%d" % (str_code, epoch),
                                      total=len(data_loader),
                                      bar_format="{l_bar}{r_bar}") 
        
        if self.embed_method not in ['PCA', 'COPY']:
            if train:
                self.model.train()
            else:
                self.model.eval()
        
        total_loss = 0
        total_MS = 0
        total_FS = 0
        for i, data in data_iter:
            #Store Data:
            if epoch == 0:
                if train:
                    self.dataset_list['train'].append(data)
                else:
                    self.dataset_list['test'].append(data)
                
            #Send Batched Data to device (cpu or gpu)
            data = {key: value.float().to(self.device) for key, value in data.items()}
            
            if self.embed_method in ['DAE', 'DFE']:
                #Run Data Through the Model    
                tgt_pred = self.model(src=data['src'])

                #Calculate Loss and Take a Step
                loss = self.loss_fn(tgt_pred, data['tgt'], data['norm'], train=train)

                #Track Total Loss
                total_loss += loss

            elif self.embed_method == 'TRANS':
                    
                src_pred, tgt_pred = self.model(src=data['src_masked'], src_mask=None) #tgt=data['tgt_input'], 
                                                #tgt_mask=data['tgt_mask'])
            
                #Calculate Loss and Take a Step
                loss, MSloss, FSloss = self.loss_fn(MSx=src_pred, MSy=data['src'].flatten(start_dim=1), 
                                                    FSx=tgt_pred, FSy=data['tgt'].flatten(start_dim=1), 
                                                    norm=data['norm'], MSprop=self.MSprop, train=train)

                #Track Total Loss
                total_loss += loss
                total_MS += MSloss
                total_FS += FSloss
                
            
            elif self.embed_method == 'PCA':
                #Run Data Through the Model
                if train:
                    self.model.partial_fit(data['src'].numpy())
                tgt_pred = self.model.inverse_transform(self.model.transform(data['src']))

                if train:
                    #Calculate Loss Loss
                    loss = self.criterion(torch.tensor(tgt_pred).float().contiguous(), 
                                          data['tgt'].contiguous()).item()
                else:
                    loss = self.loss_fn(torch.tensor(tgt_pred).float(), data['tgt'], data['norm']).item()
                
                #Track Total Loss
                total_loss += loss
                
            elif self.embed_method == 'COPY':
                data['src'] = data['src'][:, 23, :].unsqueeze(1).expand(-1, 24, -1)
                
                loss = self.loss_fn(data['src'], data['tgt'], data['norm'], train=False)
                
                total_loss += loss
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "total_loss": total_loss,
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))   
                
        
        if self.embed_method == 'PCA' and train:
            avg_loss = total_loss/len(data_iter)
        else:
            avg_loss = total_loss/data_loader.dataset.get_number_of_visits()
            if self.embed_method == 'TRANS':
                avg_MS = total_MS/data_loader.dataset.get_number_of_visits()
                avg_FS = total_FS/data_loader.dataset.get_number_of_visits()
                
                   
        if self.embed_method == 'TRANS':
            print("EP%d_%s, avg_loss=%f, MS_loss=%f, FS_loss=%f" % (epoch, str_code, avg_loss, avg_MS, avg_FS))
        else:
            print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss)
        
        if train:
            self.train_loss[epoch] = avg_loss
            if self.embed_method == 'TRANS': 
                self.train_MS[epoch] = avg_MS
                self.train_FS[epoch] = avg_FS
        else:
            self.test_loss[epoch] = avg_loss
            if self.embed_method == 'TRANS': 
                self.test_MS[epoch] = avg_MS
                self.test_FS[epoch] = avg_FS
            
        
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
    
    def save_better(self, epoch):
        """
        After all Models are saved store best performing epoch
        
        :param epoch: best epoch number
        :return: final_output_path
        """
        epoch_best = min(self.test_loss, key=self.test_loss.get)
        if epoch == epoch_best:
            self.save(epoch)
        
    def write_loss(self):
        """
        save training and test losses for each epoch as csv.
        
        :return: lossfile.csv
        """
        if self.embed_method == 'TRANS':
            with open(os.path.join(self.save_dir, 'lossfile.csv'), 'w') as lossfile:
                lossfile.write("epoch, train_loss, train_RL, train_PL, test_loss, test_RL, test_PL\n")
                i = 0
                while i in self.train_loss:
                    lossfile.write('{},{},{},{},{},{},{}\n'.format(str(i), 
                                                       str(self.train_loss[i]), str(self.train_MS[i]), str(self.train_FS[i]), 
                                                       str(self.test_loss[i]), str(self.test_MS[i]), str(self.test_FS[i])))
                    i += 1
        else:
            with open(os.path.join(self.save_dir, 'lossfile.csv'), 'w') as lossfile:
                lossfile.write("epoch, train_loss, test_loss\n")
                i = 0
                while i in self.train_loss:
                    lossfile.write('{},{},{}\n'.format(str(i), str(self.train_loss[i]), str(self.test_loss[i]))) 
                    i += 1
                
 

            
                

            
        
        
