import torch
import argparse
from source.run import train_model
from source.config import ToyModelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--log2_batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--log2_training_steps", type=int)
    parser.add_argument("--sample_kind", type=str)
    parser.add_argument("--task")
    parser.add_argument("--decay", type=float)
    parser.add_argument("--init_bias", type=float)
    parser.add_argument("--nonlinearity")
    parser.add_argument("--reg", type=float)

    args = parser.parse_args()

    k = args.k
    learning_rate = args.learning_rate
    log2_batch_size = args.log2_batch_size
    log2_training_steps = args.log2_training_steps
    sample_kind = args.sample_kind
    init_bias = args.init_bias
    nonlinearity = args.nonlinearity
    task = args.task
    decay = args.decay
    eps = args.eps
    m = args.m
    N = args.N
    reg = args.reg

    config = ToyModelConfig(
        N=args.N,
        m=args.m,
        k=args.k,
        eps=args.eps,
        batch_size=2**args.log2_batch_size,
        learning_rate=args.learning_rate,
        training_steps=2**args.log2_training_steps,
        sample_kind=args.sample_kind,
        task=args.task,
        decay=args.decay,
        init_bias=args.init_bias,
        nonlinearity=args.nonlinearity,
        reg=args.reg
    )

    data = train_model(config)

    losses, model, models, setup = data

    del setup['sampler']

    fname = f"./model3_{task}_{nonlinearity}_k_{k}_batch_{log2_batch_size}_steps_{log2_training_steps}_learning_rate_{learning_rate}_sample_{sample_kind}_init_bias_{init_bias}_decay_{decay}_eps_{eps}_m_{m}_N_{N}_reg_{reg}.pt"
    outputs = {
        'setup': setup,
        'N': N,
        'm': m,
        'k':k,
        'log2_batch_size': log2_batch_size,
        'learning_rate': learning_rate,
        'eps': eps,
        'task': task,
        'reg': reg,
        'decay': decay,
        'nonlinearity': nonlinearity,
        'initial_bias': init_bias,
        'log2_training_steps': log2_training_steps,
        'sample_kind': sample_kind,
        'losses': losses,
        'final_model': model.state_dict(),
        'log2_spaced_models': list(m.state_dict() for m in models),
    }
    torch.save(outputs, fname)

if __name__ == "__main__":
    main()