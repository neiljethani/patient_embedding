import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model.model import make_model
from .utils import NoamOpt
from .utils import LossCompute

import os
import shutil
import arrow
import pickle

import tqdm

class ClassificationTrainer():
    def __init__(self, train_dataloader: DataLoader, 
                 test_dataloader: DataLoader = None, 
                 model = None, embed_model = None,
                 embed_method: str = 'RAW', dropout: float = 0.1,
                 lr: float = 0, betas = (0.9, 0.98), eps=1e-9, 
                 factor: int = 2, warmup: int = 4000, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param model: Model to use, otherwise make_model is used
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param embed_mode: Model to use to embed data
        :param embed_method: String of Embedding Type Used
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param eps: Adam optimizer eps
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        # Set Embedding Model
        self.embed_model = embed_model
        
        # Set Embedding Method
        self.embed_method = embed_method
        
        #Initialize Model
        if model is not None:
            self.model = model.float()
        if embed_method != 'RAW':
            self.model, d_model = make_model(dropout=dropout, embed_method = embed_method, embed_model = embed_model)
        else:
            self.model, d_model = make_model(d_input = self.train_data.dataset.get_input_dim(), 
                                    dropout=dropout, embed_method = embed_method, 
                                    embed_model = embed_model)
        self.model = self.model.float()
        
        #Set Optimizer and Scheduler
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
        self.optim_schedule = NoamOpt(model_size = d_model, 
                                      factor = factor, warmup = warmup, optimizer = self.optim)
        
        #Set Loss criterion + Function + Proportion of Masked Sequence Prediction in Loss
        self.criterion = nn.BCELoss()
        self.loss_fn = LossCompute(self.criterion, self.optim_schedule)
        
        #Set Log Frequency
        self.log_freq = log_freq
        
        #Create save dir
        self.save_dir = os.path.join('/work/MIMIC/models/in_hospital_mortality/' + embed_method, 
                                     arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.best_dir = os.path.join(self.save_dir, "best")
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
               
        #Initialize Loses
        self.train_loss = {}
        self.test_loss = {}
        
    def iteration(self, epoch, data_loader, train = True):
        
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
                
            #Predict Classifcation
            y_pred = self.model(data['X'])
            
            #Calculate Loss
            loss = self.loss_fn(y_pred, data['y'], train=train)
            
            #Track Total Loss
            total_loss += loss
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": total_loss/(i+1),
                "loss": loss
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        
        avg_loss = total_loss/len(data_iter)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss)
        
        if train:
            self.train_loss[epoch] = avg_loss
        else:
            self.test_loss[epoch] = avg_loss
            
        
    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = os.path.join(self.save_dir, "{}_IHM.ep{}".format(self.embed_method, epoch))
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False) 
        
    def save_best(self):
        """
        After all Models are saved store best performing epoch
        
        :param epoch: best epoch number
        :return: final_output_path
        """
        epoch = min(self.test_loss, key=self.test_loss.get)
        src_path = os.path.join(self.save_dir, "{}_IHM.ep{}".format(self.embed_method, epoch))
        best_path = os.path.join(self.best_dir, "{}_IHM.ep{}".format(self.embed_method, epoch))
        
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
            
            
            
            
        
        
        
        