import pytest 

import torch
import torch.nn as nn
from source.activations import SoLU
from source.sampling import sample_vectors_equal, sample_vectors_power_law, make_random_embedder

from source.utils import get_activation_from_string, get_sampling_function_from_string, get_output_embedder, get_output_dim

def test_get_activation_from_string():
    
    x_range = torch.linspace(-10, 10, 100)
    for x in x_range:
        assert get_activation_from_string('ReLU')(x) == nn.ReLU()(x)
        assert get_activation_from_string('GeLU')(x) == nn.GELU()(x)
        assert get_activation_from_string('SoLU')(x) == SoLU()(x)

    with pytest.raises(ValueError):
        get_activation_from_string('invalid')

def test_get_sampling_function_from_string():
    assert get_sampling_function_from_string('equal') == sample_vectors_equal
    assert get_sampling_function_from_string('power_law') == sample_vectors_power_law
    with pytest.raises(ValueError):
        get_sampling_function_from_string('invalid')

def test_get_output_embedder():
    assert get_output_embedder('random_proj', 10, 5).shape == make_random_embedder(10,5).shape
    assert get_output_embedder('autoencoder', 10, 5) == None

def test_get_output_dim():
    assert get_output_dim('random_proj', 10, 5) == 5
    assert get_output_dim('autoencoder', 10, 5) == 10
