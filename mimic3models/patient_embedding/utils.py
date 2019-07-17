from __future__ import absolute_import
from __future__ import print_function

from mimic3models import common_utils
import numpy as np
import os


#Edditing load_data Function to Get ONLY last t hours --> Split into X and Y (Post Normalization)
def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N=1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0][-int(t):] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array([X[-int(t):-int(t/2)] for (X,t) in zip(data, ts)]), 
                  np.array([X[-int(t/2):] for (X,t) in zip(data, ts)]))
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
            
def add_transformer_arguments(parser):
    """
    arguments for trainging embedding transformer.
    """
    
    parser.add_argument("-l", "--layers", type=int, default=3, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=24, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=12, help="dataloader worker size")
    
    parser.add_argument("-mp", "--mask_loss_prop", type=int, default=0.5, help="proportion of mask sequence loss in total loss")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=0.0, help="learning rate of adam")
    parser.add_argument("--adam_eps", type=float, default=1e-9, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="adam second beta value")
    parser.add_argument("--factor", type=int, default=2, help="scheduling factor")
    parser.add_argument("--warmup", type=int, default=4000, help="iterations of warmup for scheduler")
    
    parser.add_argument('--timestep', type=float, default=1.0,
                        help="fixed timestep used in the dataset")
    parser.add_argument('--imputation', type=str, default='previous')
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--size_coef', type=float, default=4.0)
    parser.add_argument('--normalizer_state', type=str, default=None,
                        help='Path to a state file of a normalizer. Leave none if you want to '
                             'use one of the provided ones.')
    
    
