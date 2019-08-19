import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import shutil
import arrow
import pickle
import random
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.utils import resample
import scikitplot

import bootstrapped.bootstrap as bs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import tqdm


class EmbeddingVisualizer():
    def __init__(self,  output_dir,  
                 embed_model: str = None, task: str = None,
                 dataloader: DataLoader = None, embed_method: str = 'RAW',
                 with_cuda: bool = True, cuda_devices=None):
        """
        :param dataloader: dataset data loader
        :param embed_model: Model to use to embed data
        :param embed_method: String of Embedding Type Used
        :param with_cuda: traning with cuda
        """
        
        # Setup cuda device 
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(("cuda:0" if cuda_devices == None else cuda_devices)[0] if cuda_condition else "cpu")
        
        # Setting the data loader
        self.dataloader = dataloader
        self.data ={}
        
        # Set Embedding Model
        self.embed_model = embed_model
        
        # Set Embedding Method
        self.embed_method = embed_method
        
        #Store Task
        self.task = task
        
        #Store Output Dir
        self.output_dir = output_dir
        
        #Set model
        if embed_method != 'RAW':
            if embed_method != 'PCA':
                self.model = torch.load(embed_model).float().to(self.device)
                self.model.embed = True
            else:
                self.model = pickle.load(open(embed_model, 'rb'))
            
        #Initialize TSNE
        self.tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300, learning_rate=50)
        
        #Denote is tsne has been fit
        self.fitted = False
        
        #Initialize result dataframe
        self.results = pd.DataFrame()
     
    
    def set_data(self, dataloader):
        self.dataloader = dataloader
    
    
    def set_task(self, task):
        self.task = task
        
        #Set and Create Save Directory
        self.save_dir = os.path.join(self.output_dir, self.embed_method, 
                                     self.task, arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
     
    
    def fit(self):
        if not self.fitted:
            #Fit and Transform Loaded Data
            tsne_results = self.tsne.fit_transform(self.data['x'])
            tsne_results = pd.DataFrame(tsne_results)
            tsne_results.columns = ['Dimension 1', 'Dimension 2']

            #Add to Results
            self.results = pd.concat([self.results, tsne_results], axis=1)

            #Denote that tsne has been fit
            self.fitted = True
        
        
    def load_data(self):
        for i, data in enumerate(self.dataloader):
            if i != 0:
                break
            data = {key: value.float().to(self.device) for key, value in data.items()}
            if not self.data: 
                if self.embed_method != 'RAW':
                    if self.embed_method != 'PCA':
                        x = self.model(src=data['X'])
                    else:
                        x = self.model.embedding(data['X'].detach().cpu().numpy())
                else:
                    x = data['X']
                self.data['x'] = x.detach().cpu().numpy()            
            self.data[self.task] = data['y'].cpu().numpy()
            
        self.results[self.task] = self.data[self.task]
        self.results[self.task] = self.results[self.task].apply(
            lambda x: 'negative' if x==0.0 else 'positive')
        
        
    def plot(self):
        if self.fitted:
            plt.figure(figsize=(10,10))
            sns.set_palette(["#d9f0a3", "#800026"])
            sns.scatterplot(
                x='Dimension 1', y='Dimension 2',
                hue=self.task,
                style = self.task,
                data=self.results,
                legend="full",
                alpha = 0.3
            )
            plt.title('t-SNE: {} Embedding for {}'.format(self.embed_method, self.task))
            plt.savefig(os.path.join(self.save_dir, '{}.png'.format(self.task)))
            
        else:
            print('First Fit t-SNE')
            

class ClassificationVisualizer():
    def __init__(self,  output_dir,  task: str = None,
                 dataloader: DataLoader = None, embed_method: str = 'RAW',
                 with_cuda: bool = True, cuda_devices=None):
        
        """
        :param dataloader: dataset data loader
        :param embed_model: Model to use to embed data
        :param embed_method: String of Embedding Type Used
        :param with_cuda: traning with cuda
        """
        
        # Setup cuda device 
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device(("cuda:0" if cuda_devices == None else cuda_devices)[0] if cuda_condition else "cpu")
        
        # Setting the data loader
        self.dataloader = dataloader
        
        # Set Embedding Method
        self.embed_method = embed_method
        
        #Store Task
        self.task = task
        
        #Store Output Dir
        self.output_dir = output_dir
        
        #Initilialize Model
        self.model = None
            
        #Intitaliaze Predictions and Targets
        self.predictions = []
        self.targets = []
        
        #Store if Predcited
        self.predicted = False
        
        
    def set_data(self, dataloader):
        self.dataloader = dataloader
    
    
    def set_task(self, task):
        self.task = task
        
        #Set and Create Save Directory
        self.save_dir = os.path.join(self.output_dir, self.embed_method, 
                                     self.task, arrow.now().format('YYYY-MM-DD'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def set_model(self, model):
        #Set model
        self.model = torch.load(model).float().to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
            
    def predict(self):
        if not self.predicted:
            data_iter = tqdm.tqdm(enumerate(self.dataloader),
                                  total=len(self.dataloader),
                                  bar_format="{l_bar}{r_bar}")

            for i, data in data_iter:
                data = {key: (value.float().to(self.device) if type(value) is not list \
                              else torch.Tensor([int(x) for x in value]).float().to(self.device)) for key, value in data.items()}

                #Predict Classifcation
                y_pred = self.model(data['X'])

                #Store All Predictions and Targets
                self.predictions.append(y_pred)
                self.targets.append(data['y'])
                
            #Calulate Metrics
            self.predictions = torch.cat(self.predictions, 0).data.cpu().numpy()
            self.targets = torch.cat(self.targets, 0).data.cpu().numpy()
            
            if len(self.predictions.shape) == 1:
                self.predictions = np.stack([1 - self.predictions, self.predictions]).transpose((1, 0))
            self.targets = np.array(self.targets).astype(int)
                
            self.predicted = True
            
    def plot_curves(self):
        #RO Curve
        scikitplot.metrics.plot_roc(self.targets, self.predictions, 
                                    title='ROC: {} Embedding for {} '.format(self.embed_method, self.task),
                                    classes_to_plot = 1,
                                    plot_micro=False, plot_macro=False, 
                                    figsize=(10,10), cmap='nipy_spectral', 
                                    title_fontsize='large', text_fontsize='medium')
        plt.savefig(os.path.join(self.save_dir, 'ROC_{}.png'.format(self.task)))
        
        #PR Curve
        scikitplot.metrics.plot_precision_recall(self.targets, self.predictions,  
                                                 title='PRC: {} Embedding for {} '.format(self.embed_method, self.task),
                                                 classes_to_plot = 1,
                                                 plot_micro=False,
                                                 figsize=(10,10), cmap='nipy_spectral', 
                                                 title_fontsize='large', text_fontsize='medium')
        plt.savefig(os.path.join(self.save_dir, 'PRC_{}.png'.format(self.task)))
        
    def generate_metrics(self):
        scores = {'auroc':[], 'auprc':[], 'acc':[], 'f1':[]}
        for i in range(200):
            #Boostrap Predictions and Targets
            targets, predictions = resample(self.targets, self.predictions)
            #Return Metrics on Bootstapped Data
            scores['auroc'].append(metrics.roc_auc_score(targets, predictions[:, 1]))
            
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(targets, predictions[:, 1])
            scores['auprc'].append(metrics.auc(recalls, precisions))
            
            scores['acc'].append(metrics.accuracy_score(targets, predictions[:, 1]>.5))
            
            scores['f1'].append(max_f1(precisions, recalls))
            
        #Calculate Statisitcs about Bootstrpped Scores
        results = {'auroc':[], 'auprc':[], 'acc':[], 'f1':[]}
        for metric, value in scores.items():
            results[metric] = calculate_ci(value)
            
        results = pd.DataFrame(results)
        
        results.to_csv(os.path.join(self.save_dir,  'BM_{}.csv'.format(self.task)))
        
        self.metrics = results
            
              
def calculate_ci(x, alpha = 0.95):
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(x, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(x, p))
    mean = np.mean(x)
    
    return [mean, lower, upper]
    
def max_f1(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return max(f1)
        
        
        
        
        
        
        