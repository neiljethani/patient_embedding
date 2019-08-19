from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.patient_embedding import utils
from mimic3models.preprocessing import DiscretizerContinuous, Normalizer
from mimic3models import common_utils

from mimic3models.pytorch_models.classification.train.train import ClassificationTrainer

from torch.utils.data import DataLoader

from mimic3models.visualization.utils import EmbeddingVisualizer

import torch

import random
random.seed(49297)


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser()
utils.add_transformer_arguments(parser)
parser.add_argument('--data_dir', type=str, default= '/home/neil.jethani/patient_embedding/data', help='base data DIRECTORY')
parser.add_argument('--output_dir', type=str, default= '/home/neil.jethani/patient_embedding/vis', help='Directory relative which all output files are stored')
parser.add_argument('--embed_method', type=str, default = 'TRANS', help='Embedding Model to use (TRANS, DAE, PCA, RAW)')
parser.add_argument('--embed_model', type=str, help='Path to Embedding model to Use')
args = parser.parse_args()
print(args)

while args.embed_method not in ['TRANS', 'DAE', 'PCA', 'RAW', 'DFE']:
    args.embed_method = input('Enter Embedding Method (TRANS, DAE, PCA, RAW, DFE): ')
    
#Function to Create DataLoader    
def create_data_loader(data, subset=False):
    # Build readers, discretizers, normalizers
    print("Creating Data File Reader")
    #Using val_test set for visualization
    data_reader = Reader(dataset_dir=os.path.join(data, 'val_test'),
                         listfile=os.path.join(data, 'val_test', 'listfile.csv'),
                         period_length=24.0)
    
    #For Hourly Task we need to limit the amount of data for visualization
    if subset:
        print("limiting data")
        data_reader.limit_data(10)

    print("Initializing Discretizer and Normalizer")
    discretizer = DiscretizerContinuous(timestep=1.0,
                                        store_masks=False,
                                        impute_strategy='previous',
                                        start_time='zero')

    discretizer_header = discretizer.transform(data_reader.read_example(0)["X"])[1]
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ptemb_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)


    #Create Dataset + DataLoader
    print("Building Dataset")
    data_dataset = ClassDataset(reader=data_reader, discretizer=discretizer, 
                                normalizer=normalizer, return_name=False, 
                                embed_method=args.embed_method)
    

    print("Building DataLoader")
    data_loader = DataLoader(data_dataset, batch_size=len(data_dataset), shuffle=False, num_workers=args.num_workers)
    
    return data_loader


for task_type in ['day', 'hourly']:
    if task_type == 'day':
        from mimic3benchmark.readers import DayReader as Reader
        from mimic3models.pytorch_models.classification.dataset.utils import ClassificationDataset as ClassDataset
        
        #Initalizer Embedding Visualizer Class
        DayEmbeddingVis = EmbeddingVisualizer(output_dir = args.output_dir, embed_model = args.embed_model, 
                                              embed_method = args.embed_method, 
                                              with_cuda=args.with_cuda, cuda_devices=args.cuda_devices)
        
        for task in ['in_hospital_mortality', 'extended_length_of_stay']:
            print("Visualizing {}".format(task))
            #Set Data Loader and Task
            data = os.path.join(args.data_dir, task)
            dataloader = create_data_loader(data)
            DayEmbeddingVis.set_data(dataloader)
            DayEmbeddingVis.set_task(task)
            
            #Load Data
            DayEmbeddingVis.load_data()
            
            #Fit t-SNE
            DayEmbeddingVis.fit()
            
            #Plot t-SNE and Save
            DayEmbeddingVis.plot()
            
    else:
        from mimic3benchmark.readers import HourlyReader as Reader
        from mimic3models.pytorch_models.classification.dataset.utils import HourlyClassificationDataset as ClassDataset
        
        #Initalizer Embedding Visualizer Class
        HourlyEmbeddingVis = EmbeddingVisualizer(output_dir = args.output_dir, embed_model = args.embed_model, 
                                                 embed_method = args.embed_method, 
                                                 with_cuda=args.with_cuda, cuda_devices=args.cuda_devices)
        
        for task in ['decompensation', 'discharge']:
            print("Visualizing {}".format(task))
            #Set Data Loader and Task
            data = os.path.join(args.data_dir, task)
            HourlyEmbeddingVis.set_data(create_data_loader(data, True))
            HourlyEmbeddingVis.set_task(task)
            
            #Load Data
            HourlyEmbeddingVis.load_data()
            
            #Fit t-SNE
            HourlyEmbeddingVis.fit()
            
            #Plot t-SNE and Save
            HourlyEmbeddingVis.plot()
            
            





    


