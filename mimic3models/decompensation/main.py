from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.patient_embedding import utils
from mimic3benchmark.readers import HourlyReader

from mimic3models.preprocessing import DiscretizerContinuous, Normalizer
from mimic3models import common_utils

from mimic3models.pytorch_models.classification.dataset.utils import HourlyClassificationDataset
from mimic3models.pytorch_models.classification.train.train import ClassificationTrainer

from torch.utils.data import DataLoader

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser()
utils.add_transformer_arguments(parser)
parser.add_argument('--data', type=str, help='Path to the data of patient embedding task',
                    default=os.path.join(os.path.dirname(__file__), '../../../MIMIC/processed_data/in_hospital_mortality/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--embed_method', type=str, default = 'TRANS', help='Embedding Model to use (TRANS, DAE, PCA, RAW)')
parser.add_argument('--embed_model', type=str, help='Path to Embedding model to Use')
parser.add_argument('--tune', type=bool, default = False, help='tunes pretrained model')
parser.add_argument('--enable_tune', dest='tune', action='store_true')
parser.add_argument('--percent_data', type=int, default = 100, help="amount of data to use for training")
args = parser.parse_args()
print(args)

while args.embed_method not in ['TRANS', 'DAE', 'PCA', 'RAW', 'DFE']:
    args.embed_method = input('Enter Embedding Method (TRANS, DAE, PCA, RAW): ')
    

# Build readers, discretizers, normalizers
print("Creating Data File Reader")
train_reader = HourlyReader(dataset_dir=os.path.join(args.data, 'val'), 
                                    listfile=os.path.join(args.data, 'val', 'listfile.csv'), 
                                    period_length=24.0)

val_reader = HourlyReader(dataset_dir=os.path.join(args.data, 'val_test'),
                                  listfile=os.path.join(args.data, 'val_test', 'listfile.csv'),
                                  period_length=24.0)

#Limit the Percent of Data to use for training
if args.percent_data != 100:
    train_reader.limit_data(args.percent_data)

print("Initializing Discretizer and Normalizer")
discretizer = DiscretizerContinuous(timestep=1.0,
                                    store_masks=False,
                                    impute_strategy='previous',
                                    start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1]
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ptemb_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'ihm'

#Create Dataset + DataLoader
print("Building Dataset")
train_dataset = HourlyClassificationDataset(reader=train_reader, discretizer=discretizer, 
                                      normalizer=normalizer, return_name=False, 
                                      embed_method=args.embed_method)
val_dataset = HourlyClassificationDataset(reader=val_reader, discretizer=discretizer, 
                                    normalizer=normalizer, return_name=False, 
                                    embed_method=args.embed_method)
print("Building DataLoader")
trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


#Train Classifier
print("Creating Trainer")
trainer = ClassificationTrainer(task = 'DEC', model = None, output_dir = args.output_dir,
                                train_dataloader=trainLoader, test_dataloader=valLoader, 
                                embed_method = args.embed_method, embed_model = args.embed_model,
                                lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps = args.adam_eps,
                                factor = args.factor, warmup = args.factor, 
                                with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,
                                percent_data=args.percent_data)
    
print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    if valLoader is not None:
        trainer.test(epoch)
    trainer.save_better(epoch, args.performance_threshold)
    trainer.write_loss()
trainer.save_best()
trainer.write_loss()

if args.tune:
    print("Tunning Start")
    trainer.tune()
    for epoch in range(args.epochs, args.epochs+100):
        trainer.train(epoch)
        if valLoader is not None:
            trainer.test(epoch)
        trainer.save_better(epoch)
        trainer.write_loss()
    trainer.save_best()
    trainer.write_loss()

