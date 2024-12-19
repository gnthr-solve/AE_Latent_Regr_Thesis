
import os
import tempfile
import torch
import ray
import logging

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.train import Checkpoint, CheckpointConfig

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
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Loss, Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name, PeriodicSaveCallback, GlobalBestModelSaver

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def deep_regr_procedure(config, dataset):

    #report_loss_name = 'Huber'
    report_loss_name = 'L2_norm'

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    n_layers = config['n_layers']
    activation = config['activation']

    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']


    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    # Deep Regression
    regressor = DNNRegr(input_dim = input_dim, n_layers = n_layers, activation = activation)


    ###--- Losses ---###
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    regr_loss = Loss(loss_term = regr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


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

        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                #context = train.get_context()
                
                torch.save(
                    regressor.state_dict(),
                    os.path.join(temp_checkpoint_dir, f"regressor.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({report_loss_name :loss_regr.item()}, checkpoint=checkpoint)
        else:
            train.report({report_loss_name :loss_regr.item()})

        #--- Scheduler Step ---#
        scheduler.step()


    ###--- Test Loss ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'regressor': regressor},
    )

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso', loss_name = report_loss_name)
    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        RegrLossVisitor(regr_loss_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_regr = results.metrics[eval_cfg.loss_name]
    
    train.report({eval_cfg.loss_name :loss_regr})
    




"""
Optimise
-------------------------------------------------------------------------------------------------------------------------------------------
"""


if __name__=="__main__":

    ray.init()  # Initialize Ray

    ###--- Experiment Meta ---###
    experiment_name = 'deep_regr'
    #optim_metric = 'Huber'
    optim_metric = 'L2_norm'
    optim_mode = 'min'
    num_samples = 20

    storage_path = Path.cwd().parent / 'ray_results'

    #--- Dataset Meta ---#
    dataset_kind = 'key'
    normaliser_kind = 'min_max'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]

    #--- Results Directory ---#
    dir_name = f'{experiment_name}_{dataset_kind}_{normaliser_kind}' if normaliser_kind is not 'None' else f'{experiment_name}_{dataset_kind}'
    results_dir = Path(f'./results/{dir_name}/')
    os.makedirs(results_dir, exist_ok=True)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()
    

    ###--- Run Config ---###
    save_results_callback = PeriodicSaveCallback(
        save_frequency = 5, 
        experiment_name = experiment_name, 
        tracked_metrics=[optim_metric],
        results_dir=results_dir,
    )

    global_best_model_callback = GlobalBestModelSaver(
        tracked_metric = optim_metric,   
        mode = optim_mode,              
        cleanup_frequency = 10,       
        experiment_name = experiment_name,
        results_dir = results_dir,
    )

    checkpoint_cfg = CheckpointConfig(
        num_to_keep = 1, 
        checkpoint_score_attribute = optim_metric, 
        checkpoint_score_order = optim_mode
    )


    ###--- Searchspace ---###
    search_space = {
        'epochs': tune.randint(lower=2, upper = 20),
        'batch_size': tune.randint(lower=20, upper = 200),
        'n_layers': tune.randint(lower=2, upper = 15),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    }

    
    ###--- Tune Config ---###
    #search_alg = BayesOptSearch()
    #search_alg = HyperOptSearch()
    search_alg = OptunaSearch()

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    
    ###--- Setup and Run Optimisation ---###
    tuner = tune.Tuner(
        tune.with_parameters(deep_regr_procedure, dataset = dataset),
        tune_config=tune.TuneConfig(
            search_alg = search_alg,
            metric = optim_metric,
            mode = optim_mode,
            num_samples = num_samples,
            trial_dirname_creator = custom_trial_dir_name,
        ),
        run_config = train.RunConfig(
            name = experiment_name,
            storage_path = storage_path,
            checkpoint_config = checkpoint_cfg,
            callbacks = [save_results_callback, global_best_model_callback],
        ),
        param_space=search_space,
    )
    
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    
    results_df = results.get_dataframe()
    results_df.to_csv(results_dir / f'final_results.csv', index = False)