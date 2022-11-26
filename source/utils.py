import torch.nn as nn
from .activations import SoLU
from .sampling import sample_vectors_equal, sample_vectors_power_law, make_random_embedder

def get_activation_from_string(activation_string):
    if activation_string == 'ReLU':
        return nn.ReLU()
    elif activation_string == 'GeLU':
        return nn.GELU()
    elif activation_string == 'SoLU':
        return SoLU()
    else:
        raise ValueError('No valid activation specified. Quitting.')

def get_sampling_function_from_string(sampling_string):
    if sampling_string == 'equal':
        return sample_vectors_equal
    elif sampling_string == 'power_law':
        return sample_vectors_power_law
    else:
        raise ValueError('No valid sampling function specified. Quitting.')
    
def get_output_embedder(task, N, m):
    if task == 'random_proj':
        return make_random_embedder(N,m)
    else: 
        return None
    
def get_output_dim(task, N, m):
    if task == 'random_proj':
        return m
    else:
        return N