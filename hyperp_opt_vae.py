
import os
import sys
import torch
import ray
import logging

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path

from data_utils import DatasetBuilder, SplitSubsetFactory

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, ProductRegr, DNNRegr
from models import AE, VAE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    CompositeLossTerm,
    CompositeLossTermObs,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Loss, Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def VAE_iso_training_procedure(config, dataset):
    
    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    n_layers_e = config['n_layers_e']
    n_layers_d = config['n_layers_d']
    activation = config['activation']
    beta = config['beta']

    ae_lr = config['ae_lr']
    scheduler_gamma = config['scheduler_gamma']

    
    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Model ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)
    kld_term = Weigh(GaussianAnaKLDiv(), weight = beta)
    
    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}

    
    train_loss = Loss(CompositeLossTerm(loss_terms))
    test_reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = ae_lr)
    scheduler = ExponentialLR(optimizer, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = train_loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )
    
    training_procedure()


    ###--- Test Loss ---###
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'joint': test_dataset},
        models = {'AE_model': model},
    )

    eval_cfg = EvalConfig(data_key = 'joint', output_name = 'ae_iso', mode = 'iso', loss_name = 'rel_L2_loss')

    ae_output_visitor = VAEOutputVisitor(eval_cfg = eval_cfg)
    
    visitors = [
        ae_output_visitor,
        ReconstrLossVisitor(test_reconstr_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results

    train.report({eval_cfg.loss_name: results.metrics[eval_cfg.loss_name]})


"""
Optimise
-------------------------------------------------------------------------------------------------------------------------------------------
"""


if __name__=="__main__":

    ray.init()  # Initialize Ray

    ###--- Dataset ---###
    dataset_kind = 'key'
    normaliser_kind = 'min_max'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]

    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()


    ###--- Run Config ---###
    experiment_name = 'VAE_iso_procedure'
    storage_path = Path.cwd().parent / 'ray_results'


    ###--- Searchspace ---###
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'latent_dim': tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'n_layers_e': tune.choice([3, 4, 5, 6, 7, 8]),
        'n_layers_d': tune.choice([3, 4, 5, 6, 7, 8]),
        'beta': tune.uniform(0, 100),
        'ae_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    }

    
    ###--- Tune Config ---###
    optim_metric = 'Huber'
    num_samples = 1000
    #search_alg = BayesOptSearch()
    #search_alg = HyperOptSearch()
    search_alg = OptunaSearch()

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)


    ###--- Setup and Run Optimisation ---###
    tuner = tune.Tuner(
        tune.with_parameters(VAE_iso_training_procedure, dataset = dataset),
        tune_config=tune.TuneConfig(
            search_alg = search_alg,
            metric = optim_metric,
            mode = "min",
            num_samples = num_samples,
            trial_dirname_creator = custom_trial_dir_name,
        ),
        run_config = train.RunConfig(
            name = experiment_name,
            storage_path = storage_path,
        ),
        param_space=search_space,
    )
    
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

    results_df = results.get_dataframe()
    results_df.to_csv(f'./results/{experiment_name}.csv', index = False)