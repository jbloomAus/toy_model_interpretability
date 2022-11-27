import torch
from source.run import monosemanticity_runner
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Test!
# monosemanticity_runner(
#     decay=0.03,
#     initial_bias=-1,
#     sample_kind="power_law",
#     log2_training_steps=3,
#     sweep_values= [0.001,0.003],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     nonlinearity='GeLU',
#     file_name='test',
#     device = device
# )

# # LR1
# # for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
# monosemanticity_runner(
#     sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     file_name='lr1',
#     device = device
# )

# # LR2 (power law)
# # [0.001,0.003,0.005,0.007,0.01,0.03]
# monosemanticity_runner(
#     sample_kind="power_law",
#     sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     file_name='lr2',
#     device = device
# )

# # LR3 (negative bias and decay)
# # [0.001,0.003,0.005,0.007,0.01,0.03]
# monosemanticity_runner(
#     decay=0.003,
#     initial_bias=-1,
#     sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     file_name='lr3',
#     device = device
# )

# # B1
# eps = 2**-4
# bias = np.linspace(-2,1,11)
# monosemanticity_runner(
#     eps = 2**-4,
#     decay=0.03,
#     learning_rate=0.003,
#     sweep_values= bias,
#     sweep_var='initial_bias',
#     output_dir='jermyn_2022_data',
#     file_name='b1',
#     device = device
# )

# # B2
# monosemanticity_runner(
#     eps = 2**-5,
#     decay=0.003,
#     learning_rate=0.003,
#     sweep_values= bias,
#     sweep_var='initial_bias',
#     output_dir='jermyn_2022_data',
#     file_name='b2',
#     device = device
# )

# # B3
# monosemanticity_runner(
#     eps = 2**-6,
#     decay=0.003,
#     learning_rate=0.003,
#     sweep_values= bias,
#     sweep_var='initial_bias',
#     output_dir='jermyn_2022_data',
#     file_name='b3'
# )
# # B4
# monosemanticity_runner(
#     eps = 2**-7,
#     decay=0.003,
#     learning_rate=0.003,
#     sweep_values= bias,
#     sweep_var='initial_bias',
#     output_dir='jermyn_2022_data',
#     file_name='b4',
#     device = device
# )
# # B5 
# monosemanticity_runner(
#     eps = 2**-8,
#     decay=0.003,
#     sweep_values= bias,
#     sweep_var='initial_bias',
#     output_dir='jermyn_2022_data',
#     file_name='b5',
#     device = device
# )
# # LR4 (negative bias and decay and power law)
# # [0.001,0.003,0.005,0.007,0.01,0.03]
# monosemanticity_runner(
#     decay=0.03,
#     initial_bias=-1,
#     sample_kind="power_law",
#     sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     file_name='lr4',
#     device = device
# )

# # B32 # not sure how to deal with this one yet...
# bias = np.linspace(-2,1,11)
# lr = [0.001, 0.05 ,0.01]
# monosemanticity_runner(
#     decay=0.03,
#     initial_bias=-1,
#     sample_kind="power_law",
#     sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
#     sweep_var='learning_rate',
#     output_dir='jermyn_2022_data',
#     nonlinearity='GeLU',
#     file_name='g3',
#     device = device
# )

# # E1
# eps = 2**(-1*np.linspace(1, 5, 11))
# # -1 bias
# # decay rate 0.03

# # E2
# eps = 2**(-1*np.linspace(1, 5, 11))
# # -1 bias
# # decay rate 0.01

# # E3
# eps = 2**(-1*np.linspace(1, 5, 11))
# # -1 bias
# # decay rate 0.003

# # E4
# eps = 2**(-1*np.linspace(1, 5, 11))
# # -1 bias
# # decay rate 0.001


# K0 
k = [2**i for i in range(4, 12)]
# 0 bias
# decay rate 0
monosemanticity_runner(
    decay=0,
    initial_bias=0,
    learning_rate=0.007,
    sweep_values= [2**i for i in range(4, 11)],
    sweep_var='k',
    output_dir='jermyn_2022_data',
    file_name='K0',
    device = device
)
# K1
# -1 bias
# decay rate 0.03
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    learning_rate=0.007,
    sweep_values= k,
    sweep_var='k',
    output_dir='jermyn_2022_data',
    file_name='K1',
    device = device
)

# K2
# -1 bias
# decay rate 0.03
# power law
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    learning_rate=0.007,
    sample_kind="power_law",
    sweep_values= k,
    sweep_var='k',
    output_dir='jermyn_2022_data',
    file_name='K2',
    device = device
)

# # RG1 
# reg =  [10**-i for i in range(1, 8)]
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    learning_rate=0.005,
    sweep_values= [10**-i for i in range(1, 8)],
    sweep_var='reg',
    output_dir='jermyn_2022_data',
    file_name='RG1',
    device = device
)

# # RP1
# # same as LR1 but with reprojector task 
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    task="random_proj",
    sample_kind="power_law",
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='jermyn_2022_data',
    file_name='RP1',
    device = device
)

# # LR5
# # same as LR1 but with abs task, 2048 k 
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    k = 2048,
    task="random_proj",
    sample_kind="abs",
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='jermyn_2022_data',
    file_name='LR5',
    device = device
)


# # D1 Variable decay rate
# decay = [10**-i for i in range(1, 8)]
monosemanticity_runner(
    decay=0.03,
    initial_bias=-1,
    k = 2048,
    learning_rate=0.007,
    task="random_proj",
    sample_kind="abs",
    sweep_values= [10**-i for i in range(1, 8)],
    sweep_var='decay',
    output_dir='jermyn_2022_data',
    file_name='D1',
    device = device
)
