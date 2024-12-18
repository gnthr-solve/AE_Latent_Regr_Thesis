
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

def linear_regr_procedure(config, dataset):

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']
    

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    # Linear Regression
    regressor = LinearRegr(latent_dim = input_dim)


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(loss_term = regr_loss_term)
    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    for epoch in range(epochs):
        
        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            y_hat_batch = regressor(X_batch)

            loss_regr = regr_loss(
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_regr.backward()

            optimiser.step()


        scheduler.step()

    
    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    
    regr_test_ds = test_subsets['labelled']

    test_indices = regr_test_ds.indices
    X_test_l = dataset.X_data[test_indices]
    y_test_l = dataset.y_data[test_indices]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():
        y_test_l_hat = regressor(X_test_l)

        loss_regr = regr_loss_test(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    train.report({'Huber' :loss_regr})

    



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
    experiment_name = 'linear_regression_procedure'
    storage_path = Path.cwd().parent / 'ray_results'


    ###--- Searchspace ---###
    search_space = {
        'epochs': tune.randint(lower=2, upper = 200),
        'batch_size': tune.randint(lower=20, upper = 200),
        'regr_lr': tune.loguniform(1e-4, 1e-2),
        'scheduler_gamma': tune.uniform(0.5, 1),
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
        tune.with_parameters(linear_regr_procedure, dataset = dataset),
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