import torch 

@torch.jit.script
def loss_func(batch_size, outputs, vectors):
    loss = torch.sum((outputs - vectors)**2) / batch_size
    return loss

@torch.jit.script
def abs_loss_func(batch_size, outputs, vectors):
    loss = torch.sum((outputs - torch.abs(vectors))**2) / batch_size
    return loss
