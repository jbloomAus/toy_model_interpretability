import torch
import numpy as np
from .loss import loss_func, abs_loss_func

from typing import Callable

@torch.jit.script
def sample_vectors_power_law(N, eps, batch_size, embedder, device: str = 'cpu'):
    '''
    Generates random uniform vectors in a tensor of shape (N,batch_size)
    with sparsity 1-eps. These are returned as v.
    
    Applies embedding matrix to v to produce a low-dimensional embedding,
    returned as x.    
    '''
    v = torch.rand((int(batch_size), int(N)), device=device)

    compare = 1. / torch.arange(1,int(N)+1,device=device)**1.1
    compare *= N * eps / torch.sum(compare)
    compare[compare >= 1] = 1

    sparsity = torch.bernoulli(compare.repeat(int(batch_size),1))
                
    v *= sparsity
    x = torch.matmul(v,embedder.T) # Embeds features in a low-dimensional space

    return v, x

@torch.jit.script
def sample_vectors_equal(N, eps, batch_size, embedder, device: str = 'cpu'):
    '''
    Generates random uniform vectors in a tensor of shape (N,batch_size)
    with sparsity 1-eps. These are returned as v.
    
    Applies embedding matrix to v to produce a low-dimensional embedding,
    returned as x.    
    '''
    v = torch.rand((int(batch_size), int(N)), device=device)
    
    compare = eps * torch.ones((int(batch_size), int(N)), device=device)
    sparsity = torch.bernoulli(compare)
            
    v *= sparsity
    x = torch.matmul(v,embedder.T) # Embeds features in a low-dimensional space

    return v, x


def make_random_embedder(N,m, device = 'cpu'):
    matrix = np.random.randn(N,m) # Make a random matrix that's (N,m)
    u,s,v = np.linalg.svd(matrix, full_matrices=False)
    # Now u is a matrix (N,m) with orthogonal columns and nearly-orthogonal rows
    # Normalize the rows of u
    u /= (np.sum(u**2,axis=1)**0.5)[:,np.newaxis]
    t = torch.tensor(u.T, requires_grad=False, device=device, dtype=torch.float)
    return t


def task_sampler_generator(task: str, base_sampler: Callable, output_embedder: torch.Tensor = None):
    '''
    Returns a function that generates a sampler for a given task.
    Also returns the loss function for the task.
    '''

    if task == 'autoencoder':
        l_func = loss_func
        sample_vectors = base_sampler
    elif task == 'random_proj':
        if output_embedder is None:
            raise ValueError('No output embedder specified for random projection task.')
        l_func = loss_func
        sample_vectors = get_random_sampler(base_sampler, output_embedder)
    elif task == 'abs':
        l_func = abs_loss_func
        # I need to cut eps in half to make this equivalent density.
        # Different samples have different sparse choices so doubles the density.
        sample_vectors = get_abs_sampler(base_sampler)
    else:
        raise ValueError('No valid task specified. Quitting.')

    return sample_vectors, l_func

def get_random_sampler(base_sampler, output_embedder):
    def random_sampler(N, eps, batch_size, fixed_embedder):
        v,i = base_sampler(N, eps, batch_size, fixed_embedder)
        v = torch.matmul(v, output_embedder.T)
        return v,i
    return random_sampler

def get_abs_sampler(base_sampler):
    def abs_sampler(N, eps, batch_size, fixed_embedder):
        v1,i1 = base_sampler(N, eps / 2, batch_size, fixed_embedder)
        v2,i2 = base_sampler(N, eps / 2, batch_size, fixed_embedder)
        return v1 - v2, i1 - i2
    return abs_sampler