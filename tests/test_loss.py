# test sampling.py

import torch 
import pytest
from source.loss import loss_func, abs_loss_func

def test_loss_func():
    batch_size = torch.tensor(2**12)
    outputs = torch.rand(batch_size, 2**12)
    vectors = torch.rand(batch_size, 2**12)
    loss = loss_func(batch_size, outputs, vectors)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    # points if you can explain why this is the case
    torch.testing.assert_close(loss, torch.tensor(680, dtype=torch.float32), rtol=0.01, atol=0)

def test_abs_loss_func():
    batch_size = torch.tensor(2**12)
    outputs = torch.rand(batch_size, 2**12)
    vectors = torch.rand(batch_size, 2**12) 
    loss = abs_loss_func(batch_size, outputs, vectors)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    # points if you can explain why this is the case
    torch.testing.assert_close(loss, torch.tensor(680, dtype=torch.float32), rtol=0.01, atol=0)
