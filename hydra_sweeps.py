
import hydra.errors
import torch
import pandas as pd
import numpy as np
import hydra

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


from pathlib import Path
from tqdm import tqdm

from data_utils import DatasetBuilder, SplitSubsetFactory

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, ProductRegr
from models import AE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    Loss,
    CompositeLossTerm,
    CompositeLossTermObs,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
)

import os
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="ae_iso_cfg_test")
def train_AE_iso_hydra(cfg: DictConfig):

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    
    n_layers_e = cfg.n_layers_e
    n_layers_d = cfg.n_layers_d

    learning_rate = cfg.learning_rate
    scheduler_gamma = cfg.scheduler_gamma


    ###--- DataLoader ---###
    train_dataset = subset_factory.retrieve(kind='train', combine=True)
    test_dataset = subset_factory.retrieve(kind='test', combine=True)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    ###--- Models AE ---###
    input_dim = dataset.X_dim - 1

    # encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e)
    # decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d)

    # model = AE(encoder = encoder, decoder = decoder)

    #--- Models NaiveVAE ---#
    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d)

    #model = NaiveVAE_LogVar(encoder = encoder, decoder = decoder)
    model = NaiveVAE_LogSigma(encoder = encoder, decoder = decoder)
    #model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)

    #initialize_weights(model)

    
    ###--- Observers ---###
    n_iterations = len(dataloader)
    model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations, model = model)
    #loss_observer = LossObserver(n_epochs = epochs, n_iterations = n_iterations)


    ###--- Loss ---###
    #reconstr_term = AEAdapter(LpNorm(p = 2))
    reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    loss_terms = {'Reconstruction': reconstr_term}
    reconstr_loss = Loss(CompositeLossTerm(**loss_terms))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = learning_rate)
    scheduler = ExponentialLR(optimizer, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = reconstr_loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )

    # training_procedure.register_observers(
    #     #latent_observer, 
    #     #model_observer,
    # )
    
    training_procedure()


    ###--- Test Observers ---###
    #loss_observer.plot_results()
    #latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    evaluation = Evaluation(
        dataset=dataset,
        subsets={'test': test_dataset},
        models={'AE_model': model},
    )
    eval_cfg = EvalConfig(data_key='test', output_name='ae_iso', mode='iso', loss_name='rel_L2_loss')
    visitors = [
        AEOutputVisitor(eval_cfg=eval_cfg),
        ReconstrLossVisitor(reconstr_term, eval_cfg=eval_cfg),
    ]
    evaluation.accept_sequence(visitors=visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[eval_cfg.loss_name]

    return {
        eval_cfg.loss_name: loss_reconstr,
        'ae_model': model,
    }
    



@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="vae_iso_cfg_test")
def train_VAE_iso_hydra(cfg: DictConfig):

    
    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    
    n_layers_e = cfg.n_layers_e
    n_layers_d = cfg.n_layers_d

    learning_rate = cfg.learning_rate
    scheduler_gamma = cfg.scheduler_gamma
    beta = cfg.beta
    
    ###--- DataLoader ---###
    train_dataset = subset_factory.retrieve(kind='train', combine=True)
    test_dataset = subset_factory.retrieve(kind='test', combine=True)
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    dataset_size = len(train_dataset)

    latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size = dataset_size,
        batch_size = batch_size,
        members = ['Log-Likelihood', 'KL-Divergence'],
        name = 'VAE Loss',
        aggregated = True,
    )


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    kld_term = Weigh(GaussianAnaKLDiv(), weight=beta)
    #kld_term = GaussianMCKLDiv()

    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    #loss = Loss(CompositeLossTerm(**loss_terms))
    loss = Loss(CompositeLossTerm(observer = loss_observer, **loss_terms))

    test_reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = learning_rate)
    scheduler = ExponentialLR(optimizer, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure.register_observers(
        latent_observer, 
        #model_observer,
    )
    
    training_procedure()


    ###--- Test Observers ---###
    #loss_observer.plot_results()
    #latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))



    ###--- Test Loss ---###
    evaluation = Evaluation(
        dataset=dataset,
        subsets={'test': test_dataset},
        models={'AE_model': model},
    )
    eval_cfg = EvalConfig(data_key='test', output_name='vae_iso', mode='iso', loss_name='rel_L2_loss')
    reconstr_term = AEAdapter(RelativeLpNorm(p=2))
    visitors = [
        VAEOutputVisitor(eval_cfg=eval_cfg),
        ReconstrLossVisitor(reconstr_term, eval_cfg=eval_cfg),
    ]
    evaluation.accept_sequence(visitors=visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[eval_cfg.loss_name]

    return {
        eval_cfg.loss_name: loss_reconstr,
        'vae_model': model,
    }




@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="joint_nvae_regr_cfg_sweep")
def joint_epoch_shared_layer(cfg: DictConfig):

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    
    n_layers = cfg.n_layers

    encoder_lr = cfg.encoder_lr
    decoder_lr = cfg.decoder_lr
    regr_lr = cfg.regr_lr
    scheduler_gamma = cfg.scheduler_gamma

    ete_regr_weight = cfg.ete_regr_weight
    
    
    train_subsets = subset_factory.retrieve(kind = 'train')
    test_subsets = subset_factory.retrieve(kind='test')

    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers)

    ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight= 1 - ete_regr_weight), 
        'Regression Term': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    
    ete_loss = Loss(CompositeLossTerm(**ete_loss_terms))
    reconstr_loss = Loss(loss_term = reconstr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    training_procedure = JointEpochTrainingProcedure(
        ae_train_dataloader = dataloader_ae,
        regr_train_dataloader = dataloader_regr,
        ae_model = ae_model,
        regr_model = regressor,
        ae_loss = reconstr_loss,
        ete_loss = ete_loss, 
        optimizer = optimiser,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure()


    ###--- Test Loss ---###
    evaluation = Evaluation(
        dataset=dataset,
        subsets=test_subsets,
        models={'AE_model': ae_model, 'regressor': regressor},
    )

    eval_cfg_reconstr = EvalConfig(data_key='unlabelled', output_name='vae_iso', mode='iso', loss_name='rel_L2_loss')
    eval_cfg_regr = EvalConfig(data_key='labelled', output_name='vae_regr', mode='composed', loss_name='Huber_loss')

    visitors = [
        AEOutputVisitor(eval_cfg=eval_cfg_reconstr),
        ReconstrLossVisitor(reconstr_loss_term, eval_cfg=eval_cfg_reconstr),
        AEOutputVisitor(eval_cfg=eval_cfg_regr),
        ReconstrLossVisitor(regr_loss_term, eval_cfg=eval_cfg_regr),
    ]
    evaluation.accept_sequence(visitors=visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[eval_cfg_reconstr.loss_name]
    loss_regr = results.metrics[eval_cfg_regr.loss_name]

    return {
        eval_cfg_reconstr.loss_name: loss_reconstr,
        eval_cfg_regr.loss_name: loss_regr,
        'vae_model': ae_model,
        'regressor': regressor,
    }




@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="baseline_regr_cfg_sweep")
def train_baseline(cfg: DictConfig):

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size

    regr_lr = cfg.regr_lr
    scheduler_gamma = cfg.scheduler_gamma


    ###--- DataLoader ---###
    train_subsets = subset_factory.retrieve(kind = 'train')
    test_subsets = subset_factory.retrieve(kind='test')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    regressor = LinearRegr(latent_dim = input_dim)


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
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
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'regressor': regressor},
    )

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso', loss_name = 'Huber')
    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        RegrLossVisitor(regr_loss_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_regr = results.metrics[eval_cfg.loss_name]
    
    return {
        'loss_regr': loss_regr.item(),
        'regressor': regressor,
    }




"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":
    
    ###--- Device ---###
    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Dataset ---###
    kind = 'key'
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
    dataset_builder = DatasetBuilder(
        kind = kind,
        normaliser = normaliser,
        #exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    

    ###--- Setup and calculate results ---###
    #train_AE_iso_hydra()
    #train_VAE_iso_hydra()
    joint_epoch_shared_layer()
    #train_baseline()

    