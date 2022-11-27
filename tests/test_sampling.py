# test sampling.py

import torch 
import pytest
from source.sampling import sample_vectors_power_law, sample_vectors_equal, make_random_embedder

def test_sample_vectors_power_law():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**12)
    embedder = make_random_embedder(N, 2)
    v, x = sample_vectors_power_law(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, N)
    assert x.shape == (batch_size, 2)
    # check correct sparsity

    #not as narrow a dist as equal sampling
    torch.testing.assert_close((v == 0).to(torch.float32).mean(),1-eps, rtol=0, atol=0.05)

    # check embedding is accurate
    torch.testing.assert_close(x, torch.matmul(v, embedder.T))

def test_sample_vectors_equal():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**12)
    embedder = make_random_embedder(N, 2)
    v, x = sample_vectors_equal(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, N)
    assert x.shape == (batch_size, 2)

    # check correct sparsity
    torch.testing.assert_close((v == 0).to(torch.float32).mean(),1-eps, rtol=0.0001, atol=0.0001)

    # check embedding is accurate
    torch.testing.assert_close(x, torch.matmul(v, embedder.T))

def test_make_random_embedder():
    N = torch.tensor(100)
    m = torch.tensor(2)
    embedder = make_random_embedder(N, m)
    assert embedder.shape == (m, N)
    assert torch.allclose(torch.sum(embedder**2, dim=0).T, torch.ones(N))
