import torch
import numpy as np

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
