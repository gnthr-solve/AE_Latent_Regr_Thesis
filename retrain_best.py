
import os
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

from helper_tools.setup import create_eval_metric, create_normaliser


METRICS = ['L2_norm','Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm', 'Rel_L1-norm', 'L2_norm_reconstr']
TRAINING_PARAMS = [
    'timestamp', 
    'time_total_s', 
    'epochs', 
    'batch_size', 
    'regr_lr',
    'encoder_lr', 
    'decoder_lr', 
    'scheduler_gamma',
    'ete_regr_weight'
]

EVAL_METRICS = ['Rel_L2-norm','L1-norm']


"""
Retrain Linear
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
Retrain DNN
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrain_DNN_regr(results_dir: Path, experiment_name: str, data_kind: str, normaliser_kind: str):

    ###--- Paths ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'


    ###--- Load Results, identify best ---###
    results_df = pd.read_csv(results_path, low_memory = False)

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

    regressor = DNNRegr(
        input_dim = input_dim,
        output_dim = 2,
        **best_model_params,
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


    ###--- Save Model ---###
    torch.save(
        regressor.state_dict(),
        os.path.join(experiment_dir, f"regressor.pt"),
    )


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
        f"DNN Regression:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {metrics['L2_norm']}\n"
    )





"""
Retrain AE Linear
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrain_ae_linear(
        storage_dir: Path, 
        data_kind: str, 
        normaliser_kind: str,
        best_training_params: dict,
        best_model_params: dict,
    ):


    ###--- Meta ---###
    epochs = int(best_training_params['epochs'])
    batch_size = int(best_training_params['batch_size'])

    encoder_lr = best_training_params['encoder_lr']
    decoder_lr = best_training_params['decoder_lr']
    regr_lr = best_training_params['regr_lr']
    scheduler_gamma = best_training_params['scheduler_gamma']

    ete_regr_weight: float = best_training_params['ete_regr_weight']


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
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size=0.9)
    train_subsets = subset_factory.retrieve(kind='train')
    
    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    latent_dim = best_model_params['latent_dim']
    
    n_layers = best_model_params['n_layers']
    activation = best_model_params['activation']
    
    encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    
    ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = LinearRegr(latent_dim = latent_dim)
    

    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    ae_loss = Loss(loss_term = reconstr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]
            print(X_batch[:10])
            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)
            print(Z_batch[:10])
            loss_ae = ae_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
            )

            #--- Backward Pass ---#
            loss_ae.backward()

            optimiser.step()
            
        
        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)
            y_hat_batch = regressor(Z_batch)

            reconstr_component = reconstr_loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch).mean()
            regr_component = regr_loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch).mean()

            loss_ete_weighted = (1 - ete_regr_weight) * reconstr_component + ete_regr_weight * regr_component

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser.step()


        scheduler.step()


    ###--- Save Model ---###
    # torch.save(ae_model.state_dict(), os.path.join(storage_dir, f"ae_model.pt"))
    # torch.save(regressor.state_dict(), os.path.join(storage_dir, f"regressor.pt"))


    ###--- Evaluation ---###
    ae_model.eval()
    regressor.eval()
    test_subsets = subset_factory.retrieve(kind='test')

    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )


    regr_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
    }

    ae_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm_reconstr', 'Rel_L2-norm_reconstr', 'L1-norm_reconstr', 'Rel_L1-norm_reconstr']
    }
    
    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    print(metrics)

   



def retrain_ae_linear_loop(
        storage_dir: Path, 
        data_kind: str, 
        normaliser_kind: str,
        best_training_params: dict,
        best_model_params: dict,
        num_trials: int,
    ):


    ###--- Meta ---###
    epochs = int(best_training_params['epochs'])
    batch_size = int(best_training_params['batch_size'])

    encoder_lr = best_training_params['encoder_lr']
    decoder_lr = best_training_params['decoder_lr']
    regr_lr = best_training_params['regr_lr']
    scheduler_gamma = best_training_params['scheduler_gamma']

    ete_regr_weight: float = best_training_params['ete_regr_weight']


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
        subset_factory = SplitSubsetFactory(dataset = dataset, train_size=0.9)
        train_subsets = subset_factory.retrieve(kind='train')
        
        ae_train_ds = train_subsets['unlabelled']
        regr_train_ds = train_subsets['labelled']


        ###--- DataLoader ---###
        dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
        dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


        ###--- Models ---###
        input_dim = dataset.X_dim - 1
        latent_dim = best_model_params['latent_dim']
        
        n_layers = best_model_params['n_layers']
        activation = best_model_params['activation']
        
        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        
        ae_model = AE(encoder = encoder, decoder = decoder)
        regressor = LinearRegr(latent_dim = latent_dim)
        

        ###--- Losses ---###
        reconstr_loss_term = AEAdapter(LpNorm(p = 2))
        #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

        #regr_loss_term = RegrAdapter(Huber(delta = 1))
        regr_loss_term = RegrAdapter(LpNorm(p = 2))

        ae_loss = Loss(loss_term = reconstr_loss_term)
        

        ###--- Optimizer & Scheduler ---###
        optimiser = Adam([
            {'params': encoder.parameters(), 'lr': encoder_lr},
            {'params': decoder.parameters(), 'lr': decoder_lr},
            {'params': regressor.parameters(), 'lr': regr_lr},
        ])

        scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


        ###--- Training Procedure ---###
        pbar = tqdm(range(epochs))

        for epoch in pbar:
            
            ###--- Training Loop AE---###
            for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
                
                X_batch = X_batch[:, 1:]

                #--- Forward Pass ---#
                optimiser.zero_grad()
                
                Z_batch, X_hat_batch = ae_model(X_batch)

                loss_ae = ae_loss(
                    X_batch = X_batch,
                    X_hat_batch = X_hat_batch,
                )

                #--- Backward Pass ---#
                loss_ae.backward()

                optimiser.step()


            ###--- Training Loop End-To-End ---###
            for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
                
                X_batch = X_batch[:, 1:]
                y_batch = y_batch[:, 1:]

                #--- Forward Pass ---#
                optimiser.zero_grad()
                
                Z_batch, X_hat_batch = ae_model(X_batch)
                y_hat_batch = regressor(Z_batch)

                reconstr_component = reconstr_loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch).mean()
                regr_component = regr_loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch).mean()

                loss_ete_weighted = (1 - ete_regr_weight) * reconstr_component + ete_regr_weight * regr_component

                #--- Backward Pass ---#
                loss_ete_weighted.backward()

                optimiser.step()


            scheduler.step()


        ###--- Evaluation ---###
        ae_model.eval()
        regressor.eval()
        test_subsets = subset_factory.retrieve(kind='test')

        evaluation = Evaluation(
            dataset = dataset,
            subsets = test_subsets,
            models = {'AE_model': ae_model,'regressor': regressor},
        )


        regr_eval_metrics = {
            name: create_eval_metric(name)
            for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
        }

        ae_eval_metrics = {
            name: create_eval_metric(name)
            for name in ['L2-norm_reconstr', 'Rel_L2-norm_reconstr', 'L1-norm_reconstr', 'Rel_L1-norm_reconstr']
        }
        
        eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
        eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
    
        visitors = [
            AEOutputVisitor(eval_cfg = eval_cfg_comp),
            RegrOutputVisitor(eval_cfg = eval_cfg_comp),
            LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

            AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
            LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
        ]

        evaluation.accept_sequence(visitors = visitors)
        results = evaluation.results
        losses = results.losses
        metrics = results.metrics

        test_loss = metrics['L2-norm']
        print(
            f"Retrain {i}:\n"
            f"---------------------------------------------------------------\n"
            f"Avg. Loss on labelled testing subset: {test_loss}\n"
            f"---------------------------------------------------------------\n"
            f"Metrics: \n{metrics}\n"
            f"---------------------------------------------------------------\n"
        )

        if test_loss < best_loss:

            best_loss = test_loss

            ###--- Save Model ---###
            torch.save(ae_model.state_dict(), os.path.join(storage_dir, f"ae_model.pt"))
            torch.save(regressor.state_dict(), os.path.join(storage_dir, f"regressor.pt"))

   



"""
Call Specific
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrain_linear_model():

    results_dir = Path('./results/')

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




def retrain_DNN_model():

    results_dir = Path('./results/')

    #data_kind = 'key'
    data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'

    experiment_name = f'deep_NN_regr_{data_kind}_{normaliser_kind}'

    retrain_DNN_regr(
        results_dir=results_dir, 
        experiment_name=experiment_name, 
        data_kind=data_kind, 
        normaliser_kind=normaliser_kind
    )




def retrain_ae_linear_model():

    results_dir = Path('./results/')

    experiment_names = [
        'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max', 
        'AE_linear_joint_epoch_max_raw',
        'AE_deep_joint_epoch_key_raw',
    ]

    data_kind = 'key'
    #data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'

    experiment_name = f'AE_linear_joint_epoch_{data_kind}_{normaliser_kind}'

    ###--- Paths Load ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'
    results_df = pd.read_csv(results_path, low_memory = False)


    ###--- Conditions ---###
    conditions = {
        'latent_2': (lambda df: df['latent_dim'] == 2),
        'latent_less_3': (lambda df: df['latent_dim'] <=3),
    }
    
    condition_choice = 'latent_2'
    condition = conditions[condition_choice]

    ###--- Filter & Identify best ---###
    storage_dir = experiment_dir / ('models_'+ condition_choice)
    os.makedirs(storage_dir, exist_ok=True)

    results_df: pd.DataFrame = results_df[condition(results_df)].copy()

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()


    ###--- Extract Params ---###
    best_result_values = {name: val for name, val in best_entry.items() if name in METRICS}
    best_training_params = {name: val for name, val in best_entry.items() if name in TRAINING_PARAMS}
    best_model_params = {name: val for name, val in best_entry.items() if name not in set([*METRICS, *TRAINING_PARAMS])}
    print(
        f'Best entry:\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best training params: \n{best_training_params}\n'
        f'-------------------------------------\n'
        f'Best model params: \n{best_model_params}\n'
        f'-------------------------------------\n'
    )

    retrain_ae_linear(
        storage_dir= storage_dir,
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
        best_training_params = best_training_params,
        best_model_params = best_model_params,
    )




def retrain_ae_linear_model_loop():

    results_dir = Path('./results/')

    experiment_names = [
        'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max', 
        'AE_linear_joint_epoch_max_raw',
        'AE_deep_joint_epoch_key_raw',
    ]

    data_kind = 'key'
    #data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'

    experiment_name = f'AE_linear_joint_epoch_{data_kind}_{normaliser_kind}'

    ###--- Paths Load ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'
    results_df = pd.read_csv(results_path, low_memory = False)


    ###--- Conditions ---###
    conditions = {
        'latent_2': (lambda df: df['latent_dim'] == 2),
        'latent_less_3': (lambda df: df['latent_dim'] <=3),
    }
    
    condition_choice = 'latent_2'
    condition = conditions[condition_choice]

    ###--- Filter & Identify best ---###
    storage_dir = experiment_dir / ('models_'+ condition_choice)
    os.makedirs(storage_dir, exist_ok=True)

    results_df: pd.DataFrame = results_df[condition(results_df)].copy()

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()


    ###--- Extract Params ---###
    best_result_values = {name: val for name, val in best_entry.items() if name in METRICS}
    best_training_params = {name: val for name, val in best_entry.items() if name in TRAINING_PARAMS}
    best_model_params = {name: val for name, val in best_entry.items() if name not in set([*METRICS, *TRAINING_PARAMS])}
    print(
        f'Best entry:\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best training params: \n{best_training_params}\n'
        f'-------------------------------------\n'
        f'Best model params: \n{best_model_params}\n'
        f'-------------------------------------\n'
    )

    retrain_ae_linear_loop(
        storage_dir= storage_dir,
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
        best_training_params = best_training_params,
        best_model_params = best_model_params,
        num_trials = 20,
    )




"""
Execution
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    #retrain_linear_model()
    #retrain_DNN_model()

    retrain_ae_linear_model()
    #retrain_ae_linear_model_loop()
    
    pass