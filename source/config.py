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
    init_bias: float # initial bias
    nonlinearity: str # activation function
    reg: float # regularization parameter