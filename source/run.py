import torch
import torch.nn as nn
from train import train

from utils import get_activation_from_string, get_sampling_function_from_string, get_output_embedder, get_output_dim
from models import get_model
from sampling import sample_vectors_equal, sample_vectors_power_law, make_random_embedder

def train_model(config, device='cpu'):
    
    # parse arguments (replace this with an input parser)
    sampler = get_sampling_function_from_string(config.sample_kind)
    activation = get_activation_from_string(config.nonlinearity)
    output_embedder = get_output_embedder(config.task, config.N, config.m)
    output_dim = get_output_dim(config.task, config.N, config.m)

    # store config
    setup = {
        'N':config.N,
        'm':config.m,
        'k':config.k,
        'batch_size':config.batch_size,
        'learning_rate':config.learning_rate,
        'eps':config.eps,
        'fixed_embedder': make_random_embedder(config.N,config.m), # I need to think about this
        'sampler':sampler,
        'task': config.task,
        'decay': config.decay,
        'reg': config.reg,
        'output_embedder': output_embedder,
        'output_dim': output_dim,
        'activation': activation,
        'init_bias': config.init_bias
    }

    # get model, set bias
    model = get_model(config.m, config.k, output_dim, activation, device)
    model = set_bias_mean(model, config.init_bias)

    # train model
    losses, model, models = train(setup, model, config.training_steps)

    return losses, model, models, setup

def set_bias_mean(model, mean):
    state = model.state_dict()
    state['0.bias'] += mean
    model.load_state_dict(state)
    return model