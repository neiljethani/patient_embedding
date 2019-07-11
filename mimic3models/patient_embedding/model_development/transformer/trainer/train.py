import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

from ..model.model import make_model
from .utils.optim_schedule import NoamOpt
from .utils.loss import MedEmbedLoss

from ..model.utils.general_utils import subsequent_mask

import os
import arrow

import tqdm

class TrandformerTrainer():
    """
    TrandformerTrainer trains the MedEmbed Transformer model with two Embedding Objectives.

        1. Masked Sequence Model: Task #1: Predict Masked (Zeroed Out) Sequence Elements Within Encoding
        2. Future Sequence Prediction : Predict the Next N Hours During Decoding
    """
    
    def __init__(self, train_dataloader: DataLoader, model = None, 
                 layers: int = 3, heads: int = 4, dropout: float = 0.1, MSprop: float = 0.5,
                 test_dataloader: DataLoader = None, 
                 lr: float = 0, betas = (0.9, 0.98), eps=1e-9, 
                 factor: int = 2, warmup: int = 4000, 
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param make_model: Function to Build Model
        :param layers: Number of Layers to use for Each Encoding Step, Decoding = 2xlayers
        :param heads: Number of Heads to Use for Multihead Attention
        :param dropout: Dropout to use
        :param MSprop: Proportion of Loss Total that Masked Sequence Prediction Accounts for
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
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
        
        #Initialize Model
        if model is not None:
            self.model = model.float()
        print(self.train_data.dataset.get_input_dim())
        print(self.train_data.dataset.get_seq_length())
        self.model = make_model(d_input = self.train_data.dataset.get_input_dim(), 
                                N=layers, h=heads, max_len= self.train_data.dataset.get_seq_length(), 
                                dropout=dropout).float()
        
        #Set Optimizer and Scheduler
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
        self.optim_schedule = NoamOpt(model_size = self.model.src_embed[0].d_model, 
                                      factor = factor, warmup = warmup, optimizer = self.optim)
        
        #Set Loss criterion + Function + Proportion of Masked Sequence Prediction in Loss
        self.criterion = nn.MSELoss()
        self.loss_fn = MedEmbedLoss(self.model.generator, self.criterion, self.optim_schedule)
        self.MSprop = MSprop
        
        #Set Log Frequency
        self.log_freq = log_freq
        
        #Create save dir
        self.save_dir = os.path.join('/work/MIMIC/models/patient_embedding/transformer', 
                                     arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    
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
            tgt_mask = Variable(subsequent_mask(data['tgt_input'].size(1)).type_as(data['tgt_input'].data))
            src_mask = torch.sum(data['src'] == 0, dim = 2).unsqueeze(-2)
            print(data['src'].shape)
            src_pred, tgt_pred = self.model.forward(src=data['src_masked'], tgt=data['tgt_input'], 
                                                    src_mask=src_mask, tgt_mask=tgt_mask)
            
            #Calculate Loss and Take a Step
            loss = self.loss_fn(MSx=src_pred, MSy=data['src'], MSmask=data['mask'], 
                                FSx=tgt_pred, FSy=data['tgt'], 
                                norm=data['norm'], MSprop=self.MSprop, train=train)
            
            #Track Total Loss
            total_loss += loss
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "total_loss": total_loss,
                "loss": loss
            }
            
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), total_loss/len(data_loader.dataset))
        
            
    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = self.save_dir + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False) 
            
            
        