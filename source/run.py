import torch
import os
import numpy as np
from .train import train
from .utils import get_activation_from_string, get_sampling_function_from_string, get_output_embedder, get_output_dim
from .models import get_model
from .sampling import make_random_embedder
from .config import ToyModelConfig, create_sweep_configs

import logging
from asyncio.log import logger
import sys

# get debug level from environment variable
debug_level = os.environ.get('DEBUG_LEVEL', 'WARN')

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=debug_level,
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("run")

def monosemanticity_runner(
        N = 512,
        m = 64,
        k = 1024,
        eps = 1/64.0,
        log2_batch_size = 7,
        learning_rate = 0.03,
        log2_training_steps = 15,
        sample_kind = 'equal',
        task = 'autoencoder',
        decay = 0.0,
        initial_bias = 0.0,
        nonlinearity = 'ReLU',
        reg = 0.0,
        output_dir = './',
        sweep_var = None,
        sweep_values = None,
        file_name = None,
        device = 'cpu'
    ):
    assert sample_kind in ['equal', 'power_law']
    assert task in ['abs', 'autoencoder', 'random_proj']
    assert nonlinearity in ['ReLU', 'GeLU', 'SoLU']

    base_config = ToyModelConfig( 
        N = N,
        m = m,
        k = k,
        eps = eps,
        batch_size = 2**log2_batch_size,
        learning_rate = learning_rate,
        training_steps = 2**log2_training_steps,
        sample_kind = sample_kind,
        task = task,
        decay = decay,
        initial_bias = initial_bias,
        nonlinearity = nonlinearity,
        reg = reg,
        device = device
    )

    logger.info(f"Running with config: {base_config}")
    
    if sweep_var is not None:
        assert sweep_values is not None
        configs = create_sweep_configs(base_config, sweep_var, sweep_values)
    else:
        configs = [base_config]

    outputs = [train_model(config, config.device) for config in configs]
    repacked_outputs = [repack_model_outputs(config, output[0], output[1], output[2], output[3]) for config, output in zip(configs, outputs)]
    
    # delete output['setup']['sampler'] for output in repacked_outputs (as it can't be pickled)
    for output in repacked_outputs:
        del output['setup']['sampler']

    sweep_results = {
        'outputs': repacked_outputs,
        'sweep_var': sweep_var,
        'sweep_values': sweep_values
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(sweep_results, os.path.join(output_dir, file_name) + '.pt')

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
        'initial_bias': config.initial_bias
    }

    # get model, set bias
    model = get_model(config.m, config.k, output_dim, activation, device)
    model = set_bias_mean(model, config.initial_bias)

    # train model
    losses, model, models = train(setup, model, config.training_steps, device = device)

    return losses, model, models, setup

def set_bias_mean(model, mean):
    state = model.state_dict()
    state['0.bias'] += mean
    model.load_state_dict(state)
    return model

# legacy (for reference)
def save_model(input_config, losses, model, models, setup, output_dir = './'):
    fname = f"model3_{input_config.task}_{input_config.nonlinearity}_k_{input_config.k}_batch_{np.log2(input_config.batch_size)}_steps_{np.log2(input_config.training_steps)}_learning_rate_{input_config.learning_rate}_sample_{input_config.sample_kind}_initial_bias_{input_config.initial_bias}_decay_{input_config.decay}_eps_{input_config.eps}_m_{input_config.m}_N_{input_config.N}_reg_{input_config.reg}.pt"
    fname = os.path.join(output_dir, fname)
    outputs = repack_model_outputs(input_config, losses, model, models, setup)
    torch.save(outputs, fname)

def repack_model_outputs(input_config, losses, model, models, setup):
    outputs = {
        'setup': setup,
        'N': input_config.N,
        'm': input_config.m,
        'k':input_config.k,
        'log2_batch_size': np.log2(input_config.batch_size),
        'learning_rate': input_config.learning_rate,
        'eps': input_config.eps,
        'task': input_config.task,
        'reg': input_config.reg,
        'decay': input_config.decay,
        'nonlinearity': input_config.nonlinearity,
        'initial_bias': input_config.initial_bias,
        'log2_training_steps': np.log2(input_config.training_steps),
        'sample_kind': input_config.sample_kind,
        'losses': losses,
        'final_model': model.state_dict(),
        'log2_spaced_models': list(m.state_dict() for m in models),
    }
    return outputs