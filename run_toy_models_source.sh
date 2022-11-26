python3 -m main \
    --N 512 \
    --m 64 \
    --k 1024 \
    --eps 0.015625 \
    --log2_batch_size 7 \
    --learning_rate 0.03 \
    --log2_training_steps 17 \
    --sample_kind equal \
    --task autoencoder \
    --decay 0 \
    --init_bias 0 \
    --nonlinearity ReLU \
    --reg 0 \
    --sweep_var learning_rate \
    --sweep_values 0.001 0.003 0.005 0.007 0.01 0.03\
    --file_name LR1_reduced \

