import torch
import argparse
import os
from source.run import train_model
from source.config import ToyModelConfig
import numpy as np

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

    save_model(config, losses, model, models, setup)

def save_model(input_config, losses, model, models, setup, output_dir = './'):
    fname = f"model3_{input_config.task}_{input_config.nonlinearity}_k_{input_config.k}_batch_{np.log2(input_config.batch_size)}_steps_{np.log2(input_config.training_steps)}_learning_rate_{input_config.learning_rate}_sample_{input_config.sample_kind}_init_bias_{input_config.init_bias}_decay_{input_config.decay}_eps_{input_config.eps}_m_{input_config.m}_N_{input_config.N}_reg_{input_config.reg}.pt"
    fname = os.path.join(output_dir, fname)
    outputs = repack_model_outputs(input_config, losses, model, models, setup)
    torch.save(outputs, fname)

def repack_model_outputs(input_config, losses, model, models, setup):
    outputs = {
        'setup': setup,
        'N': input_config.N,
        'm': input_config.m,
        'k':input_config.k,
        'log2_batch_size': np.log2(input_config.batch_size),
        'learning_rate': input_config.learning_rate,
        'eps': input_config.eps,
        'task': input_config.task,
        'reg': input_config.reg,
        'decay': input_config.decay,
        'nonlinearity': input_config.nonlinearity,
        'initial_bias': input_config.init_bias,
        'log2_training_steps': np.log2(input_config.training_steps),
        'sample_kind': input_config.sample_kind,
        'losses': losses,
        'final_model': model.state_dict(),
        'log2_spaced_models': list(m.state_dict() for m in models),
    }
    return outputs
    
if __name__ == "__main__":
    main()