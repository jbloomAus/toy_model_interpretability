from source.run import monosemanticity_runner
import numpy as np


# LR1
# for lr in [0.001,0.003,0.005,0.007,0.01,0.03]
monosemanticity_runner(
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='hubinger_2022_data',
    file_name='lr1'
)

# LR2 (power law)
# [0.001,0.003,0.005,0.007,0.01,0.03]
monosemanticity_runner(
    sample_kind="power_law",
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='hubinger_2022_data',
    file_name='lr2'
)
# LR3 (negative bias and decay)
# [0.001,0.003,0.005,0.007,0.01,0.03]
monosemanticity_runner(
    decay=0.003,
    init_bias=-1,
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='hubinger_2022_data',
    file_name='lr3'
)

# B1
eps = 2**-4
bias = np.linspace(-2,1,11)
monosemanticity_runner(
    eps = 2**-4,
    decay=0.03,
    sweep_values= bias,
    sweep_var='init_bias',
    output_dir='hubinger_2022_data',
    file_name='b1'
)

# B2
monosemanticity_runner(
    eps = 2**-5,
    decay=0.03,
    sweep_values= bias,
    sweep_var='init_bias',
    output_dir='hubinger_2022_data',
    file_name='b2'
)

# B3
monosemanticity_runner(
    eps = 2**-6,
    decay=0.03,
    sweep_values= bias,
    sweep_var='init_bias',
    output_dir='hubinger_2022_data',
    file_name='b3'
)
# B4
monosemanticity_runner(
    eps = 2**-7,
    decay=0.03,
    sweep_values= bias,
    sweep_var='init_bias',
    output_dir='hubinger_2022_data',
    file_name='b4'
)
# B5 
monosemanticity_runner(
    eps = 2**-8,
    decay=0.03,
    sweep_values= bias,
    sweep_var='init_bias',
    output_dir='hubinger_2022_data',
    file_name='b5'
)

# LR4 (negative bias and decay and power law)
# [0.001,0.003,0.005,0.007,0.01,0.03]
monosemanticity_runner(
    decay=0.03,
    init_bias=-1,
    sample_kind="power_law",
    sweep_values= [0.001,0.003,0.005,0.007,0.01,0.03],
    sweep_var='learning_rate',
    output_dir='hubinger_2022_data',
    file_name='lr4'
)

# # B32 # not sure how to deal with this one yet...
# bias = np.linspace(-2,1,11)
# lr = [0.001, 0.05 ,0.01]


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


# # K0 
# k = [2**i for i in range(4, 11)]
# # 0 bias
# # decay rate 0

# # K1
# k = [2**i for i in range(4, 11)]
# # -1 bias
# # decay rate 0.03

# # K2
# k = [2**i for i in range(4, 11)]
# # -1 bias
# # decay rate 0.03
# # power law

# # RG1 
# reg =  [10**-i for i in range(1, 8)]

# # RP1
# # same as LR1 but with reprojector task 

# # LR5
# # same as LR1 but with abs task, 2048 k 

# # D1 Variable decay rate
# decay = [10**-i for i in range(1, 8)]
