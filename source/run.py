import torch
import torch.nn as nn
from train import train

from utils import get_activation_from_string, get_sampling_function_from_string, get_output_embedder, get_output_dim
from models import get_model
from sampling import sample_vectors_equal, sample_vectors_power_law, make_random_embedder

def train_model(N,m,k,eps,batch_size,learning_rate,training_steps,sample_kind,init_bias,nonlinearity,task,decay,reg, device = 'cpu'):
    
    # parse arguments
    sampler = get_sampling_function_from_string(sample_kind)
    activation = get_activation_from_string(nonlinearity)
    output_embedder = get_output_embedder(task, N, m)
    output_dim = get_output_dim(task, N, m)

    # store config
    setup = {
        'N':N,
        'm':m,
        'k':k,
        'batch_size':batch_size,
        'learning_rate':learning_rate,
        'eps':eps,
        'fixed_embedder': make_random_embedder(N,m), # I need to think about this
        'sampler':sampler,
        'task': task,
        'decay': decay,
        'reg': reg,
        'output_embedder': output_embedder,
        'output_dim': output_dim,
        'activation': activation,
        'init_bias': init_bias
    }

    # get model, set bias
    model = get_model(m, k, output_dim, activation, device)
    model = set_bias_mean(model, init_bias)

    # train model
    losses, model, models = train(setup, model, training_steps)

    return losses, model, models, setup

def set_bias_mean(model, mean):
    state = model.state_dict()
    state['0.bias'] += mean
    model.load_state_dict(state)
    return model