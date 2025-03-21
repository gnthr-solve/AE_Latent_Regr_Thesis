
from ray import train, tune

from data_utils.info import exclude_columns
from .config import ExperimentConfig, DatasetConfig
from .trainables import (
    linear_regr,
    deep_regr,
    VAE_iso,
    AE_linear_joint_epoch,
    AE_linear_joint_epoch_prime,
    AE_deep_joint_epoch,
)

"""
Concrete Configs - Data
-------------------------------------------------------------------------------------------------------------------------------------------
"""
data_cfg = DatasetConfig(
    dataset_kind = 'key',
    normaliser_kind = 'min_max',
    #exclude_columns = exclude_columns,
)



"""
Concrete Configs - Linear Regression Iso/Direct
-------------------------------------------------------------------------------------------------------------------------------------------
"""
linear_regr_iso_cfg = ExperimentConfig(
    experiment_name = 'linear_regr_iso',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 1000,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.8, 1),
    },
    trainable = linear_regr,
    eval_metrics = ['Rel_L2-norm', 'L1-norm'],
    data_cfg = data_cfg,
)




"""
Concrete Configs - Deep Regression Iso/Direct
-------------------------------------------------------------------------------------------------------------------------------------------
"""
deep_NN_regr_cfg = ExperimentConfig(
    experiment_name = 'deep_NN_regr',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 1000,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'n_fixed_layers': tune.randint(lower = 3, upper = 10),
        'fixed_layer_size': tune.randint(lower = 100, upper = 300),
        'n_funnel_layers': tune.randint(lower = 2, upper = 10),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.8, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = deep_regr,
    eval_metrics = ['Rel_L2-norm', 'L1-norm'],
    data_cfg = data_cfg,
)




shallow_NN_regr_cfg = ExperimentConfig(
    experiment_name = 'shallow_NN_regr',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 2000,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'n_fixed_layers': tune.randint(lower = 1, upper = 5),
        'fixed_layer_size': tune.randint(lower = 200, upper = 600),
        'n_funnel_layers': tune.randint(lower = 2, upper = 5),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.8, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = deep_regr,
    eval_metrics = ['Rel_L2-norm', 'L1-norm'],
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
    num_samples = 2000,
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers_e': tune.randint(lower = 3, upper = 15),
        'n_layers_d': tune.randint(lower = 3, upper = 15),
        'beta': tune.uniform(0, 100),
        'ae_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.8, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = VAE_iso,
    eval_metrics = ['Rel_L2-norm_reconstr'],
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
    num_samples = 500,
    search_space = {
        'epochs': tune.randint(lower = 70, upper = 200),
        'batch_size': tune.randint(lower = 20, upper = 100),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers': tune.randint(lower = 3, upper = 10),
        'encoder_lr': tune.loguniform(1e-4, 5e-2),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 5e-2),
        'ete_regr_weight': tune.uniform(0.5, 1),
        'scheduler_gamma': tune.uniform(0.95, 1),
        'activation': tune.choice(['LeakyReLU', 'Softplus']),
    },
    trainable = AE_linear_joint_epoch,
    eval_metrics = ['Rel_L2-norm', 'L1-norm', 'L2-norm_reconstr', 'Rel_L2-norm_reconstr'],
    data_cfg = data_cfg,
    model_params = {'AE_model_type': 'AE'}
)




nvae_linear_joint_epoch_cfg = ExperimentConfig(
    experiment_name = 'NVAE_linear_joint_epoch',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 2000,
    search_space = {
        'epochs': tune.randint(lower = 2, upper = 200),
        'batch_size': tune.randint(lower = 20, upper = 200),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers': tune.randint(lower = 3, upper = 15),
        'encoder_lr': tune.loguniform(1e-4, 1e-1),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'ete_regr_weight': tune.uniform(0, 1),
        'scheduler_gamma': tune.uniform(0.8, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = AE_linear_joint_epoch,
    eval_metrics = ['Rel_L2-norm', 'L1-norm', 'L2-norm_reconstr', 'Rel_L2-norm_reconstr'],
    data_cfg = data_cfg,
    model_params = {'AE_model_type': 'NVAE'}
)




"""
Concrete Configs - AE Deep Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""
ae_deep_joint_epoch_cfg = ExperimentConfig(
    experiment_name = 'AE_deep_joint_epoch',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 500,
    search_space = {
        'epochs': tune.randint(lower = 70, upper = 200),
        'batch_size': tune.randint(lower = 20, upper = 100),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers': tune.randint(lower = 3, upper = 10),
        'n_fixed_layers': tune.randint(lower = 1, upper = 10),
        'fixed_layer_size': tune.randint(lower = 20, upper = 100),
        'n_funnel_layers': tune.randint(lower = 2, upper = 10),
        'encoder_lr': tune.loguniform(1e-4, 5e-2),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 5e-2),
        'ete_regr_weight': tune.uniform(0.5, 1),
        'scheduler_gamma': tune.uniform(0.95, 1),
        'activation': tune.choice(['LeakyReLU', 'Softplus']),
    },
    trainable = AE_deep_joint_epoch,
    eval_metrics = ['Rel_L2-norm', 'L1-norm', 'L2-norm_reconstr', 'Rel_L2-norm_reconstr'],
    data_cfg = data_cfg,
    model_params = {'AE_model_type': 'AE'}
)




nvae_deep_joint_epoch_cfg = ExperimentConfig(
    experiment_name = 'NVAE_deep_joint_epoch',
    optim_loss = 'L2_norm',
    optim_mode = 'min',
    num_samples = 2000,
    search_space = {
        'epochs': tune.randint(lower = 2, upper = 200),
        'batch_size': tune.randint(lower = 20, upper = 200),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers': tune.randint(lower = 3, upper = 10),
        'n_fixed_layers': tune.randint(lower = 1, upper = 10),
        'fixed_layer_size': tune.randint(lower = 20, upper = 100),
        'n_funnel_layers': tune.randint(lower = 2, upper = 10),
        'encoder_lr': tune.loguniform(1e-4, 1e-1),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'ete_regr_weight': tune.uniform(0, 1),
        'scheduler_gamma': tune.uniform(0.8, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    },
    trainable = AE_deep_joint_epoch,
    eval_metrics = ['Rel_L2-norm', 'L1-norm', 'L2-norm_reconstr', 'Rel_L2-norm_reconstr'],
    data_cfg = data_cfg,
    model_params = {'AE_model_type': 'NVAE'}
)




"""
Concrete Configs - AE Linear Joint Epoch EXPERIMENT
-------------------------------------------------------------------------------------------------------------------------------------------
"""
ae_linear_joint_epoch_prime_cfg = ExperimentConfig(
    experiment_name = 'AE_linear_joint_epoch_prime',
    optim_loss = 'L2-norm',
    optim_mode = 'min',
    num_samples = 3,
    search_space = {
        'epochs': tune.randint(lower = 70, upper = 100),
        'batch_size': tune.randint(lower = 50, upper = 200),
        'latent_dim': tune.randint(lower = 2, upper = 10),
        'n_layers': tune.randint(lower = 3, upper = 10),
        'encoder_lr': tune.loguniform(1e-4, 5e-2),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 5e-2),
        'ete_regr_weight': tune.uniform(0.5, 1),
        'scheduler_gamma': tune.uniform(0.95, 1),
        'activation': tune.choice(['LeakyReLU', 'Softplus']),
    },
    trainable = AE_linear_joint_epoch_prime,
    eval_metrics = ['Rel_L2-norm', 'L1-norm', 'L2-norm_reconstr', 'Rel_L2-norm_reconstr'],
    data_cfg = data_cfg,
    model_params = {'AE_model_type': 'AE'}
)


