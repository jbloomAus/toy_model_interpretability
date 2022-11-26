import torch
import torch.nn as nn

def get_model(m, k, output_dim, nonlinearity, device):

    model = torch.jit.script(
                torch.nn.Sequential(
                    nn.Linear(m, k, bias=True),
                    nonlinearity,
                    nn.Linear(k, output_dim, bias=False)
                )
        ).to(device)

    return model