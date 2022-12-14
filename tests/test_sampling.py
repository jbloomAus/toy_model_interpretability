# test sampling.py

import torch 
import pytest
from source.sampling import sample_vectors_power_law, sample_vectors_equal, make_random_embedder
from source.sampling import task_sampler_generator, get_random_sampler, get_abs_sampler

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
    m = torch.tensor(50)
    embedder = make_random_embedder(N, m)
    assert embedder.shape == (m, N)
    assert torch.allclose(torch.sum(embedder**2, dim=0).T, torch.ones(N))

    
    orthogonal_rows_matrix = (embedder @ embedder.T) /  (embedder @ embedder.T).max()
    #px.imshow(orthogonal_rows_matrix).show() # how to visualize
    torch.testing.assert_close(orthogonal_rows_matrix, torch.eye(m), rtol=0, atol=0.1)

    orthogonal_cols_matrix = (embedder.T @ embedder) /  (embedder.T @ embedder).max()
    #px.imshow(orthogonal_cols_matrix).show() # how to visualize
    torch.testing.assert_close(orthogonal_cols_matrix, torch.eye(N), rtol=0, atol=0.5)

def test_get_abs_sampler():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**12)
    embedder = make_random_embedder(N, 2)
    abs_sampler = get_abs_sampler(sample_vectors_equal)
    v, x = abs_sampler(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, N)
    assert x.shape == (batch_size, 2)

    # check correct sparsity
    torch.testing.assert_close((v == 0).to(torch.float32).mean(),1-eps, rtol=0.01, atol=0.01)

    # check embedding is accurate
    torch.testing.assert_close(x, torch.matmul(v, embedder.T))

def test_get_random_sampler():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**12)
    output_embedder = make_random_embedder(N, 2)
    fixed_embedder = output_embedder#make_random_embedder(N, 2)
    random_sampler = get_random_sampler(sample_vectors_equal, output_embedder)
    v, x = random_sampler(N, eps, batch_size, fixed_embedder)
    assert v.shape == (N, 2)
    assert x.shape == (batch_size, 2)

    # check correct sparsity
    torch.testing.assert_close((v == 0).to(torch.float32).mean(),torch.tensor(0, dtype = torch.float32), rtol=0.01, atol=0.01)

    # check embedding is accurate
    torch.testing.assert_close(x, v) # since we can't take inverses of rectangular matrices, we can cheat by making the embeddings equal

def test_task_sampler_generator_autoencoder():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**5)
    embedder = make_random_embedder(N, 2)
    sample_vectors, l_func = task_sampler_generator("autoencoder", sample_vectors_equal)
    v, x = sample_vectors(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, N)
    assert x.shape == (batch_size, 2)

def test_task_sampler_generator_random_proj():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**5)
    embedder = make_random_embedder(N, 2)
    sample_vectors, l_func = task_sampler_generator("random_proj", sample_vectors_equal, embedder)
    v, x = sample_vectors(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, 2)
    assert x.shape == (batch_size, 2)

def test_task_sampler_generator_random_proj_no_embedder():
    with pytest.raises(ValueError):
        sample_vectors, l_func = task_sampler_generator("random_proj", sample_vectors_equal)

def test_task_sampler_generator_abs():
    N = torch.tensor(2**12)
    eps = torch.tensor(0.1)
    batch_size = torch.tensor(2**5)
    embedder = make_random_embedder(N, 2)
    sample_vectors, l_func = task_sampler_generator("abs", sample_vectors_equal)
    v, x = sample_vectors(N, eps, batch_size, embedder)
    assert v.shape == (batch_size, N)
    assert x.shape == (batch_size, 2)


def test_task_sampler_generator_unkown_task():
    with pytest.raises(ValueError):
        sample_vectors, l_func = task_sampler_generator("unknown_task", sample_vectors_equal)

