import torch
import argparse
import os
from source.run import train_model
from source.config import ToyModelConfig, create_sweep_configs
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=0.015625) # 1/64
    parser.add_argument("--log2_batch_size", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--log2_training_steps", type=int, default=12)
    parser.add_argument("--sample_kind", type=str, default = 'equal')
    parser.add_argument("--task", type=str, default = 'autoencoder')
    parser.add_argument("--decay", type=float, default = 0.0)
    parser.add_argument("--init_bias", type=float, default = 0.0)
    parser.add_argument("--nonlinearity", type=str, default = 'ReLU')
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default='./')
    parser.add_argument("--sweep_var", type=str)
    parser.add_argument("--sweep_values", type=float, default=None, nargs='+')
    parser.add_argument("--file_name", type=str, default=None)
    args = parser.parse_args()

    base_config = ToyModelConfig(
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
    if args.sweep_var is not None:
        assert args.sweep_values is not None
        configs = create_sweep_configs(base_config, args.sweep_var, args.sweep_values)
    else:
        configs = [base_config]

    outputs = [train_model(config) for config in configs]
    repacked_outputs = [repack_model_outputs(config, output[0], output[1], output[2], output[3]) for config, output in zip(configs, outputs)]
    
    # delete output['setup']['sampler'] for output in repacked_outputs (as it can't be pickled)
    for output in repacked_outputs:
        del output['setup']['sampler']

    sweep_results = {
        'outputs': repacked_outputs,
        'sweep_var': args.sweep_var,
        'sweep_values': args.sweep_values
    }

    torch.save(sweep_results, os.path.join(args.output_dir, args.file_name) + '.pt')


# legacy (for reference)
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