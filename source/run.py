import torch
import torch.nn as nn
from train import train
from activations import SoLU
from models import get_model
from sampling import sample_vectors_equal, sample_vectors_power_law, make_random_embedder

def train_model(N,m,k,eps,batch_size,learning_rate,training_steps,sample_kind,init_bias,nonlinearity,task,decay,reg, device = 'cpu'):
    if sample_kind == 'equal':
        sampler = sample_vectors_equal
    elif sample_kind == 'power_law':
        sampler = sample_vectors_power_law
    else:
        print('Sample kind not recognized. Exiting.')
        exit()

    setup = {
        'N':N,
        'm':m,
        'k':k,
        'batch_size':batch_size,
        'learning_rate':learning_rate,
        'eps':eps,
        'fixed_embedder': make_random_embedder(N,m),
        'sampler':sampler,
        'task': task,
        'decay': decay,
        'reg': reg
    }

    if task == 'random_proj':
        setup['output_embedder'] = make_random_embedder(N,m)
        
    if nonlinearity == 'ReLU':
        activation = nn.ReLU()
    elif nonlinearity == 'GeLU':
        activation = nn.GELU()
    elif nonlinearity == 'SoLU':
        activation = SoLU()
    else:
        print('No valid activation specified. Quitting.')
        exit()

    if task == 'random_proj':
        output_dim = m
    else:
        output_dim = N

    model = get_model(m, k, output_dim, activation, device)

    state = model.state_dict()
    state['0.bias'] += init_bias
    model.load_state_dict(state)
                
    losses, model, models = train(setup, model, training_steps)
    return losses, model, models, setup