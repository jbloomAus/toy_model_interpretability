time python3 -m main \
    --N 512 \
    --m 64 \
    --k 1024 \
    --eps 0.015625 \
    --log2_batch_size 7 \
    --learning_rate 0.03 \
    --log2_training_steps 12 \
    --sample_kind equal \
    --task autoencoder \
    --decay 0 \
    --initial_bias 0 \
    --nonlinearity ReLU \
    --reg 0 \
    --file_name /test

time python3 -m adam_jermyn.model3 \
    --N 512 \
    --m 64 \
    --k 1024 \
    --eps 0.015625 \
    --log2_batch_size 7 \
    --learning_rate 0.03 \
    --log2_training_steps 12 \
    --sample_kind equal \
    --task autoencoder \
    --decay 0 \
    --init_bias 0 \
    --nonlinearity ReLU \
    --reg 0 \

# (toy_model_interpretability) josephbloom@Josephs-MacBook-Pro toy_model_interpretability % ./scripts/time_benchmark.sh
# 18:49:24 INFO:run: Running with config: ToyModelConfig(N=512, m=64, k=1024, eps=0.015625, batch_size=128, learning_rate=0.03, training_steps=4096, sample_kind='equal', task='autoencoder', decay=0.0, initial_bias=0.0, nonlinearity='ReLU', reg=0.0, device='cpu')
# Loss: 1.1016: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4096/4096 [08:26<00:00,  8.09it/s]
# 18:57:50 INFO:run: Finished run in 506.11509799957275 seconds

# real    8m26.646s
# user    58m23.737s
# sys     0m32.318s

# real    10m5.328s
# user    71m1.522s
# sys     0m29.619s