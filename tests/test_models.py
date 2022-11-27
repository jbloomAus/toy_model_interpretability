# test sampling.py

import torch 
import pytest
from source.models import get_model, set_bias_mean

def test_get_model():
    m = 2**4
    k = 2**6
    output_dim = 2**5
    nonlinearity = torch.nn.ReLU()
    device = torch.device("cpu")
    model = get_model(m, k, output_dim, nonlinearity, device)
    parameters = dict(model.named_parameters())
    
    # shapes of parameters
    assert parameters['0.weight'].shape == (k,m)
    assert parameters['0.bias'].shape == (k,)
    assert parameters['2.weight'].shape == (output_dim,k)

    # bias
    torch.testing.assert_close(parameters['0.weight'].mean(), torch.tensor(0.0), rtol=0, atol=0.1)
    torch.testing.assert_close(parameters['0.bias'].mean(), torch.tensor(0.0), rtol=0, atol=0.1)
    torch.testing.assert_close(parameters['2.weight'].mean(), torch.tensor(0.0), rtol=0, atol=0.1)

    # forward pass 
    assert model.forward(torch.ones(10, m)).shape == (10, output_dim)

def test_set_bias_mean():
    m = 2**4
    k = 2**6
    output_dim = 2**5
    nonlinearity = torch.nn.ReLU()
    device = torch.device("cpu")
    model = get_model(m, k, output_dim, nonlinearity, device)
    mean = 1.0
    model = set_bias_mean(model, mean)
    parameters = dict(model.named_parameters())
    torch.testing.assert_close(parameters['0.bias'].mean(), torch.tensor(mean), rtol=0, atol=0.1)