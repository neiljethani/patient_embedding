from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0,"/work/pip")

import numpy as np
import argparse
import os
import imp
import re

from mimic3models.patient_embedding import utils
from mimic3benchmark.readers import PatientEmbeddingReader

from mimic3models.preprocessing import DiscretizerContinuous, Normalizer
from mimic3models import common_utils

from mimic3models.patient_embedding.transformer.dataset.utils import PatientEmbeddingDataset
from mimic3models.patient_embedding.transformer.trainer.train import TrandformerTrainer

from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
utils.add_transformer_arguments(parser)
parser.add_argument('--target_repl_coef', type=float, default=0.0)
parser.add_argument('--data', type=str, help='Path to the data of patient embedding task',
                    default=os.path.join(os.path.dirname(__file__), '../../../MIMIC/processed_data/patient_embedding/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
args = parser.parse_args()
print(args)


target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')

# Build readers, discretizers, normalizers
print("Creating Data File Reader")
train_reader = PatientEmbeddingReader(dataset_dir=os.path.join(args.data, 'train'), 
                                      listfile=os.path.join(args.data, 'train', 'listfile.csv'), 
                                      period_length=48.0)

val_reader = PatientEmbeddingReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'train', 'listfile.csv'),
                                       period_length=48.0)

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
args_dict['task'] = 'ptemb'
args_dict['target_repl'] = target_repl

#Create Dataset + DataLoader
print("Building Dataset")
train_dataset = PatientEmbeddingDataset(reader=train_reader, discretizer=discretizer, normalizer=normalizer, return_name=False)
val_dataset = PatientEmbeddingDataset(reader=val_reader, discretizer=discretizer, normalizer=normalizer, return_name=False)
print("Building DataLoader")
trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#Train Model
print("Creating Trainer")
trainer = TrandformerTrainer(model = None, train_dataloader=trainLoader, test_dataloader=valLoader, 
                             layers = args.layers, heads = args.attn_heads, MSprop = args.mask_loss_prop,
                             lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps = args.adam_eps,
                             factor = args.factor, warmup = args.factor, 
                             with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch)
    if valLoader is not None:
        trainer.test(epoch)





