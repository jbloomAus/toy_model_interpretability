import pytest 

from source.config import ToyModelConfig, copy_config_update_var, create_sweep_configs

def test_toy_model_config_is_not_editable():
    config = ToyModelConfig(N=10, m=5, k=3, eps=0.1, batch_size=100, learning_rate=0.001, training_steps=10000, sample_kind='sample_vectors_equal', task='autoencoder', decay=0.1, initial_bias=0.0, nonlinearity='relu', reg=0.0, device='cpu')
    with pytest.raises(AttributeError):
        config.N = 20

def test_copy_config_update_var():
    config = ToyModelConfig(N=10, m=5, k=3, eps=0.1, batch_size=100, learning_rate=0.001, training_steps=10000, sample_kind='sample_vectors_equal', task='autoencoder', decay=0.1, initial_bias=0.0, nonlinearity='relu', reg=0.0, device='cpu')
    new_config = copy_config_update_var(config, 'N', 20)
    assert new_config.N == 20

def test_create_sweep_configs():
    config = ToyModelConfig(N=10, m=5, k=3, eps=0.1, batch_size=100, learning_rate=0.001, training_steps=10000, sample_kind='sample_vectors_equal', task='autoencoder', decay=0.1, initial_bias=0.0, nonlinearity='relu', reg=0.0, device='cpu')
    sweep_values = [20, 30, 40]
    sweep_var = 'N'
    configs = create_sweep_configs(config, sweep_var, sweep_values)
    assert configs[0].N == 20
    assert configs[1].N == 30
    assert configs[2].N == 40

