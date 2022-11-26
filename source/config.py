from dataclasses import dataclass

@dataclass(frozen=True)
class ToyModelConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    N: int # number of true features
    m: int # embedding dimension ("bottleneck")
    k: int # width of model
    eps: float # 1-sparsity of vectors
    batch_size: int # number of samples in a batch
    learning_rate: float # learning rate for optimizer,
    training_steps: int # number of training steps
    sample_kind: str # sample_vectors_equal,
    task: str # the corresponding samplers from autoencoder, random proj, abs
    decay: float # decay rate for bias parameter
    initial_bias: float # initial bias
    nonlinearity: str # activation function
    reg: float # regularization parameter
    device: str # the device we originally trained on

def copy_config_update_var(config, var, value):
    params = [i for i in dir(config) if not i.startswith('_')]
    config_dict = {i: config.__getattribute__(i) for i in params}
    config_dict[var] = value
    new_config = ToyModelConfig(**config_dict)
    return new_config

def create_sweep_configs(config, sweep_var, sweep_values):
    configs = []
    for sweep_value in sweep_values:
        configs.append(copy_config_update_var(config, sweep_var, sweep_value))
    return configs