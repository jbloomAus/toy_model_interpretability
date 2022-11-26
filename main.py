import torch
import argparse
import os
import numpy as np
from source.run import monosemanticity_runner

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

    monosemanticity_runner(
        N = args.N,
        m = args.m,
        k = args.k,
        eps = args.eps,
        log2_batch_size = args.log2_batch_size,
        learning_rate = args.learning_rate,
        log2_training_steps = args.log2_training_steps,
        sample_kind = args.sample_kind,
        task = args.task,
        decay = args.decay,
        init_bias = args.init_bias,
        nonlinearity = args.nonlinearity,
        reg = args.reg,
        output_dir = args.output_dir,
        sweep_var = args.sweep_var,
        sweep_values = args.sweep_values,
        file_name = args.file_name
    )


if __name__ == "__main__":
    main()