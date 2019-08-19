import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model.model import make_model
from .utils import NoamOpt
from .utils import LossCompute
from ....metrics import print_metrics_binary

import os
import shutil
import arrow
import pickle
import random
import pandas as pd

import tqdm

class ClassificationTrainer():
    def __init__(self, task, output_dir, train_dataloader: DataLoader, 
                 test_dataloader: DataLoader = None, 
                 model = None, embed_model = None,
                 embed_method: str = 'RAW', dropout: float = 0.1,
                 lr: float = 0, betas = (0.9, 0.98), eps=1e-9, 
                 factor: int = 2, warmup: int = 4000, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10,
                 percent_data: int = 100):
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
        
        # Setup cuda device 
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(("cuda:0" if cuda_devices == None else cuda_devices)[0] if cuda_condition else "cpu")
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        # Set Embedding Model
        self.embed_model = embed_model
        
        # Set Embedding Method
        self.embed_method = embed_method
        
        #Store Task
        self.task = task
        
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
        self.model.to(self.device)
        child_n = 0
        for child in self.model.children():
            print(child)
            child_n += 1
            print(child_n)
        
        #Set Optimizer and Scheduler
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
        self.optim_schedule = NoamOpt(model_size = d_model, 
                                      factor = factor, warmup = warmup, optimizer = self.optim)
        
        #Set Loss criterion + Function + Proportion of Masked Sequence Prediction in Loss
        self.criterion = nn.BCELoss()
        self.loss_fn = LossCompute(self.criterion, self.optim) #self.optim_schedule)
        
        #Set Log Frequency
        self.log_freq = log_freq
        
        #Create save dir
        if percent_data == 100:
            self.save_dir = os.path.join(output_dir, self.embed_method, 
                                         arrow.now().format('YYYY-MM-DD'))
        else:
            self.save_dir = os.path.join(output_dir, self.embed_method, str(percent_data),
                                         arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.best_dir = os.path.join(self.save_dir, "best")
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
               
        #Initialize Loses
        self.train_loss = {}
        self.test_loss = {}
        self.train_auroc = {}
        self.test_auroc = {}
        
        #Initialize Metrics
        self.metrics = []
        
        #Create Dict of Lists to Store Mini-batches
        self.dataset_list = {'train':[], 'test':[]}
        
    def iteration(self, epoch, data_loader, train = True):
        
        str_code = "train" if train else "test"
        
        #if epoch % 100 == 0 and train:
            #self.optim.lr = float(input('Enter Learning Rate: '))
            #self.loss_fn.optim = self.optim

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
        
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        predictions = []
        targets = []
        for i, data in data_iter:
            #Store Data:
            if epoch == 0:
                if train:
                    self.dataset_list['train'].append(data)
                else:
                    self.dataset_list['test'].append(data)
            
            #Send Batched Data to device (cpu or gpu)
            data = {key: (value.float().to(self.device) if type(value) is not list else torch.Tensor([int(x) for x in value]).float().to(self.device)) for key, value in data.items()}
                
            #Predict Classifcation
            y_pred = self.model(data['X'])
            
            
            #Calculate Loss
            loss = self.loss_fn(y_pred, data['y'], train=train)
            
            #Track Total Loss
            total_loss += loss
            
            #Store All Predictions and Targets
            predictions.append(y_pred)
            targets.append(data['y'])
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": total_loss/(i+1),
                "loss": loss
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        
        #Calulate Metrics
        predictions = torch.cat(predictions, 0).data.cpu().numpy()
        targets = torch.cat(targets, 0).data.cpu().numpy()
        metrics = print_metrics_binary(targets, predictions, verbose=False)
        
        #Calculate Avg Loss
        metrics['avg_loss'] = total_loss/len(data_iter)
        metrics['epoch'] = epoch
        metrics['dataset'] = str_code
        
        #Print Results
        print("EP{}_{}, avg_loss={}, accuracy = {}, AUC of ROC = {}, AUC of PRC = {}".\
              format(epoch, str_code,  metrics['avg_loss'], metrics['acc'], metrics['auroc'], metrics['auprc']))
        
        self.metrics.append(metrics)
        
        if train:
            self.train_loss[epoch] = metrics['avg_loss']
            self.train_auroc[epoch] = metrics['auroc']
        else:
            self.test_loss[epoch] = metrics['avg_loss']
            self.test_auroc[epoch] = metrics['auroc']   
        
    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = os.path.join(self.save_dir, "{}_{}.ep{}".format(self.embed_method, self.task, epoch))
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    def save_better(self, epoch, threshold=0.5):
        """
        After all Models are saved store best performing epoch
        
        :param epoch: best epoch number
        :return: final_output_path
        """
        epoch_best = max(self.test_auroc, key=self.test_auroc.get)
        if epoch == epoch_best and self.test_auroc[epoch] >= threshold:
            self.save(epoch)
    
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
        epoch = max(self.test_auroc, key=self.test_auroc.get)
        src_path = os.path.join(self.save_dir, "{}_{}.ep{}".format(self.embed_method, self.task, epoch))
        self.best_path = os.path.join(self.best_dir, "{}_{}.ep{}".format(self.embed_method, self.task, epoch))
        
        #Copy Model to Best Directory
        shutil.copyfile(src_path, self.best_path)
        
    def write_loss(self):
        """
        save training and test losses for each epoch as csv.
        
        :return: lossfile.csv
        """
        pd.DataFrame(self.metrics).to_csv(os.path.join(self.save_dir, 'lossfile.csv'))
        
    
    def tune(self):
        print(self.best_path)
        self.model = torch.load(self.best_path)
        self.model.to(self.device)
        for i, child in enumerate(self.model.children()):
            if i == 0:
                for p in child.parameters():
                    p.requires_grad = True
                    print(p)
            #elif i == 1:
                #for p in child.parameters():
                    #p.requires_grad = False
                    #print(p)
        
            
            
            
        
        
        
        