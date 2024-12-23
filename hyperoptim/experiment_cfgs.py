
from ray import train, tune

from .config import ExperimentConfig, DatasetConfig
from .trainables import (
    linear_regr,
    deep_regr,
    VAE_iso,
    AE_linear_joint_epoch,
)

"""
Concrete Configs - Data
-------------------------------------------------------------------------------------------------------------------------------------------
"""
data_cfg = DatasetConfig(
    dataset_kind = 'key',
    normaliser_kind = 'min_max',
    exclude_columns = ['Time_ptp', 'Time_ps1_ptp', 'Time_ps5_ptp', 'Time_ps9_ptp'],
)

"""
Concrete Configs - Deep Regression Iso/Direct
-------------------------------------------------------------------------------------------------------------------------------------------
"""
deep_regr_cfg = ExperimentConfig(
    experiment_name = 'deep_regr',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 20,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 10),
        'batch_size': tune.randint(lower=20, upper = 200),
        'n_layers': tune.randint(lower=2, upper = 15),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = deep_regr,
    data_cfg = data_cfg,
)



"""
Concrete Configs - AE Linear Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""
ae_linear_joint_epoch_cfg = ExperimentConfig(
    experiment_name = 'AE_linear_joint_epoch',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 20,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 10),
        'batch_size': tune.randint(lower=20, upper = 200),
        'latent_dim': tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'n_layers': tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
        'encoder_lr': tune.loguniform(1e-4, 1e-1),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'ete_regr_weight': tune.uniform(0, 1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = AE_linear_joint_epoch,
    metrics = ['L2_norm_reconstr'],
    data_cfg = data_cfg,
)



"""
Concrete Configs - Linear Regression Iso/Direct
-------------------------------------------------------------------------------------------------------------------------------------------
"""
linear_regr_iso_cfg = ExperimentConfig(
    experiment_name = 'linear_regr_iso',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 20,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 10),
        'batch_size': tune.randint(lower=20, upper = 200),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.5, 1),
    },
    trainable = linear_regr,
    data_cfg = data_cfg,
)



"""
Concrete Configs - VAE Iso
-------------------------------------------------------------------------------------------------------------------------------------------
"""
vae_iso_cfg = ExperimentConfig(
    experiment_name = 'VAE_iso',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 20,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 10),
        'batch_size': tune.randint(lower=20, upper = 200),
        'latent_dim': tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'n_layers_e': tune.choice([3, 4, 5, 6, 7, 8]),
        'n_layers_d': tune.choice([3, 4, 5, 6, 7, 8]),
        'beta': tune.uniform(0, 100),
        'ae_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = VAE_iso,
    data_cfg = data_cfg,
)