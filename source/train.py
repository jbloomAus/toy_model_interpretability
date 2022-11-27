import torch
import torch_optimizer as optim
from copy import deepcopy
from .loss import loss_func, abs_loss_func
from .sampling import task_sampler_generator
import tqdm

def train(setup, model, training_steps, device: str = 'cpu'):
    N = setup['N'] if isinstance(setup['N'], torch.Tensor) else torch.tensor(setup['N'])
    eps = setup['eps'] if isinstance(setup['eps'], torch.Tensor) else torch.tensor(setup['eps'])
    learning_rate = setup['learning_rate'] if isinstance(setup['learning_rate'], torch.Tensor) else torch.tensor(setup['learning_rate'])
    batch_size = setup['batch_size'] if isinstance(setup['batch_size'], torch.Tensor) else torch.tensor(setup['batch_size'])
    fixed_embedder = setup['fixed_embedder']
    task = setup['task']
    decay = setup['decay']
    reg = setup['reg']

    sample_vectors, l_func = task_sampler_generator(
            task, base_sampler=setup['sampler'], 
            output_embedder= setup['output_embedder']
    )

    optimizer = optim.Lamb(model.parameters(), lr=setup['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**9, eta_min=0)

    losses = []
    models = []

    # Training loop
    pbar = tqdm.tqdm(range(training_steps))
    for i in pbar:
        optimizer.zero_grad(set_to_none=True)

        vectors, inputs = sample_vectors(N, eps, batch_size, fixed_embedder)
        vectors.to(device)
        inputs.to(device)
        outputs = model.forward(inputs)
        activations = model[:2].forward(inputs)
        l = l_func(batch_size, outputs, vectors)      
        loss = l + reg*torch.sum(torch.abs(activations)) / batch_size 

        loss.backward()

        optimizer.step()
        scheduler.step()

        pbar.set_description(f'Loss: {loss.item():.4f}')

        if i < training_steps / 2:
            state = model.state_dict()
            state['0.bias'] *= (1 - decay * learning_rate)
            model.load_state_dict(state)

        if i%2**4 == 0: # Avoids wasting time on copying the scalar over
            losses.append(float(l))

        if (i & (i+1) == 0) and (i+1) != 0: # Checks if i is a power of 2
            models.append(deepcopy(model))

    return losses, model, models
