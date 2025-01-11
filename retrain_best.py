
import os
import tempfile
import torch
import pandas as pd
import logging

from pathlib import Path

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm

from data_utils import TensorDataset, SplitSubsetFactory, DatasetBuilder, get_subset_by_label_status

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, DNNRegr
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

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor, LossTermVisitor
)

from visualisation import *

from helper_tools import normalise_tensor
from helper_tools.setup import create_eval_metric, create_normaliser

METRICS = ['L2_norm','Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm', 'Rel_L1-norm']
TRAINING_PARAMS = ['timestamp', 'time_total_s', 'epochs', 'batch_size', 'regr_lr', 'scheduler_gamma']

EVAL_METRICS = ['Rel_L2-norm','L1-norm']


"""
Model retrain functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrain_linear_regr(results_dir: Path, experiment_name: str, data_kind: str, normaliser_kind: str):

    ###--- Paths ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'

    ###--- Load Results, identify best ---###
    results_df = pd.read_csv(results_path, low_memory = False)
    # drop_cols = [col for col in TRAINING_PARAMS if col in results_df.columns]
    # results_df.drop(columns = drop_cols, inplace = True)

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()
    best_result_values = {name: val for name, val in best_entry.items() if name in METRICS}
    best_training_params = {name: val for name, val in best_entry.items() if name in TRAINING_PARAMS}
    best_model_params = {name: val for name, val in best_entry.items() if name not in set([*METRICS, *TRAINING_PARAMS])}
    print(
        f'Best entry dict: \n{best_entry}\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best training params: \n{best_training_params}\n'
        f'-------------------------------------\n'
        f'Best model params: \n{best_model_params}\n'
        f'-------------------------------------\n'
    )


    ###--- Meta ---###
    epochs = int(best_training_params['epochs'])
    batch_size = int(best_training_params['batch_size'])
    
    regr_lr = best_training_params['regr_lr']
    scheduler_gamma = best_training_params['scheduler_gamma']


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")


    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    regressor = LinearRegr(
        latent_dim = input_dim,
        y_dim = 2,
    )


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


    ###--- Training Loop ---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
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


    ###--- Save Model ---###
    # torch.save(
    #     regressor.state_dict(),
    #     os.path.join(experiment_dir, f"regressor.pt"),
    # )


    ###--- Eval ---###
    regressor.eval()
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'regressor': regressor},
    )

    eval_metrics = {'L2_norm': regr_loss_term, **{name: create_eval_metric(name) for name in EVAL_METRICS}}
    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso')

    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        LossTermVisitor(loss_terms = eval_metrics, eval_cfg = eval_cfg)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    print(
        f"Regression Baseline:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {metrics['L2_norm']}\n"
    )




def retrain_linear_regr_loop(results_dir: Path, experiment_name: str, data_kind: str, normaliser_kind: str, num_trials: int):

    ###--- Paths ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'

    ###--- Load Results, identify best ---###
    results_df = pd.read_csv(results_path, low_memory = False)
    # drop_cols = [col for col in TRAINING_PARAMS if col in results_df.columns]
    # results_df.drop(columns = drop_cols, inplace = True)

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()
    best_result_values = {name: val for name, val in best_entry.items() if name in METRICS}
    best_training_params = {name: val for name, val in best_entry.items() if name in TRAINING_PARAMS}
    best_model_params = {name: val for name, val in best_entry.items() if name not in set([*METRICS, *TRAINING_PARAMS])}
    print(
        f'Best entry dict: \n{best_entry}\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best training params: \n{best_training_params}\n'
        f'-------------------------------------\n'
        f'Best model params: \n{best_model_params}\n'
        f'-------------------------------------\n'
    )


    ###--- Meta ---###
    epochs = int(best_training_params['epochs'])
    batch_size = int(best_training_params['batch_size'])
    
    regr_lr = best_training_params['regr_lr']
    scheduler_gamma = best_training_params['scheduler_gamma']


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    best_loss = torch.inf
    for i in range(num_trials):
        ###--- Dataset Split ---###
        subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
        train_subsets = subset_factory.retrieve(kind = 'train')

        regr_train_ds = train_subsets['labelled']

        dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

        regressor = LinearRegr(
            latent_dim = input_dim,
            y_dim = 2,
        )


        ###--- Losses ---###
        regr_loss_term = RegrAdapter(LpNorm(p = 2))

        regr_loss = Loss(loss_term = regr_loss_term)


        ###--- Optimizer & Scheduler ---###
        optimiser = Adam([
            {'params': regressor.parameters(), 'lr': regr_lr},
        ])

        scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


        ###--- Training Loop ---###
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            
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


        ###--- Eval ---###
        regressor.eval()
        test_datasets = subset_factory.retrieve(kind = 'test')
        
        evaluation = Evaluation(
            dataset = dataset,
            subsets = test_datasets,
            models = {'regressor': regressor},
        )

        eval_metrics = {'L2_norm': regr_loss_term, **{name: create_eval_metric(name) for name in EVAL_METRICS}}
        eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso')

        visitors = [
            RegrOutputVisitor(eval_cfg = eval_cfg),
            LossTermVisitor(loss_terms = eval_metrics, eval_cfg = eval_cfg)
        ]

        evaluation.accept_sequence(visitors = visitors)
        results = evaluation.results
        metrics = results.metrics
        test_loss = metrics['L2_norm']
        print(
            f"Regression Baseline in trial {i}:\n"
            f"---------------------------------------------------------------\n"
            f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
            f"Avg. Loss on labelled testing subset: {metrics['L2_norm']}\n"
        )

        if test_loss < best_loss:

            best_loss = test_loss

            ###--- Save Model ---###
            torch.save(
                regressor.state_dict(),
                os.path.join(experiment_dir, f"regressor.pt"),
            )
            

"""
Call Specific
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrain_model():

    results_dir = Path('./results/')

    # #data_kind = 'key'
    # data_kind = 'max'
    # normaliser_kind = 'raw'
    # #normaliser_kind = 'min_max'

    # experiment_name = f'deep_NN_regr_{data_kind}_{normaliser_kind}'

    # dnn_regr_results(
    #     results_dir=results_dir, 
    #     experiment_name=experiment_name, 
    #     data_kind=data_kind, 
    #     normaliser_kind=normaliser_kind
    # )


    #data_kind = 'key'
    data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'
    experiment_name = f'linear_regr_iso_{data_kind}_{normaliser_kind}'

    # retrain_linear_regr(
    #     results_dir=results_dir, 
    #     experiment_name=experiment_name, 
    #     data_kind=data_kind, 
    #     normaliser_kind=normaliser_kind
    # )

    retrain_linear_regr_loop(
        results_dir=results_dir, 
        experiment_name=experiment_name, 
        data_kind=data_kind, 
        normaliser_kind=normaliser_kind,
        num_trials= 10,
    )




"""
Execution
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    retrain_model()

    pass