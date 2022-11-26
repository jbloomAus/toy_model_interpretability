python3 -m main \
    --N 256 \
    --m 32 \
    --k 612 \
    --eps 0.015625 \
    --log2_batch_size 6 \
    --learning_rate 0.03 \
    --log2_training_steps 13 \
    --sample_kind equal \
    --task autoencoder \
    --decay 0 \
    --init_bias 0 \
    --nonlinearity ReLU \
    --reg 0 \
    --sweep_var learning_rate \
    --sweep_values 0.001 0.003 \
    --file_name LR1_reduced \
    --device cpu 

