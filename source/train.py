import torch
import torch_optimizer as optim
from copy import deepcopy
from .loss import loss_func, abs_loss_func
import tqdm

def train(setup, model, training_steps):
    N = setup['N'] if isinstance(setup['N'], torch.Tensor) else torch.tensor(setup['N'])
    eps = setup['eps'] if isinstance(setup['eps'], torch.Tensor) else torch.tensor(setup['eps'])
    learning_rate = setup['learning_rate'] if isinstance(setup['learning_rate'], torch.Tensor) else torch.tensor(setup['learning_rate'])
    batch_size = setup['batch_size'] if isinstance(setup['batch_size'], torch.Tensor) else torch.tensor(setup['batch_size'])
    fixed_embedder = setup['fixed_embedder']
    task = setup['task']
    decay = setup['decay']
    reg = setup['reg']

    if task == 'autoencoder':
        l_func = loss_func
        sample_vectors = setup['sampler']
    elif task == 'random_proj':
        l_func = loss_func
        def sample_vectors(N, eps, batch_size, fixed_embedder):
            v,i = setup['sampler'](N, eps, batch_size, fixed_embedder)
            v = torch.matmul(v, setup['output_embedder'].T)
            return v,i
    elif task == 'abs':
        l_func = abs_loss_func
        # I need to cut eps in half to make this equivalent density.
        # Different samples have different sparse choices so doubles the density.
        def sample_vectors(N, eps, batch_size, fixed_embedder):
            v1,i1 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
            v2,i2 = setup['sampler'](N, eps / 2, batch_size, fixed_embedder)
            return v1 - v2, i1 - i2
    else:
        print('Task not recognized. Exiting.')
        exit()

    optimizer = optim.Lamb(model.parameters(), lr=setup['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2**9, eta_min=0)

    losses = []
    models = []

    # Training loop
    pbar = tqdm.tqdm(range(training_steps))
    for i in pbar:
        optimizer.zero_grad(set_to_none=True)

        vectors, inputs = sample_vectors(N, eps, batch_size, fixed_embedder)
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
