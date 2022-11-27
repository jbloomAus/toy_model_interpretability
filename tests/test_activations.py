# test activations.py

import torch 
import pytest
from source.activations import SoLU

def test_SoLU():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = SoLU()(x)
    torch.testing.assert_allclose(y, torch.tensor([i*torch.exp(i) for i in x]))

