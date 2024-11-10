
import torch
import pandas as pd
import numpy as np
import hydra

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig


from pathlib import Path
from tqdm import tqdm

from models.encoders import (
    LinearEncoder,
)

from models.decoders import (
    LinearDecoder,
)

from models.var_encoders import VarEncoder
from models.var_decoders import VarDecoder

from models.regressors import LinearRegr
from models import AE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    Loss,
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    HuberOwn,
)

from loss.decorators import Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL

from observers import LossTermObserver, CompositeLossTermObserver, LossObserver, ModelObserver, VAELatentObserver

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser, RobustScalingNormaliser
from datasets import SplitSubsetFactory

from helper_tools import plot_loss_tensor, get_valid_batch_size, plot_training_characteristics, simple_timer
from helper_tools import DatasetBuilder



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
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


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
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]
    Z_batch_hat, X_test_hat = model(X_test)

    loss_reconst = reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)

    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'n_layers_e': n_layers_e,
        'n_layers_d': n_layers_d,
        'learning_rate': learning_rate,
        'scheduler_gamma': scheduler_gamma,
        'test_loss': loss_reconst.item(),
    }
    



@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="ae_iso_cfg_sweep")
def train_VAE_iso_hydra(cfg: DictConfig):

    
    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    
    n_layers_e = cfg.n_layers_e
    n_layers_d = cfg.n_layers_d

    learning_rate = cfg.learning_rate
    scheduler_gamma = cfg.scheduler_gamma

    
    ###--- DataLoader ---###
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs, 
        n_iterations = n_iterations,
        loss_names = ['Log-Likelihood', 'KL-Divergence'],
    )


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
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
    loss_observer.plot_results()
    latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))



    ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    Z_batch, infrm_dist_params, genm_dist_params = model(X_test)

    mu_r, _ = genm_dist_params.unbind(dim = -1)

    X_test_hat = mu_r

    loss_reconst_test = test_reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)

    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'n_layers_e': n_layers_e,
        'n_layers_d': n_layers_d,
        'learning_rate': learning_rate,
        'scheduler_gamma': scheduler_gamma,
        'test_loss': loss_reconst_test.item(),
    }




@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="joint_ae_regr_cfg_sweep")
def train_joint_epoch_procedure(cfg: DictConfig):

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    
    n_layers_e = cfg.n_layers_e
    n_layers_d = cfg.n_layers_d

    encoder_lr = cfg.encoder_lr
    decoder_lr = cfg.decoder_lr
    regr_lr = cfg.regr_lr
    scheduler_gamma = cfg.scheduler_gamma

    ete_regr_weight = cfg.ete_regr_weight
    
    
    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e)
    decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d)

    # encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e)
    # decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d)

    # #model = NaiveVAE_LogVar(encoder = encoder, decoder = decoder)
    # model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    # n_iterations_ae = len(dataloader_ae)
    # n_iterations_regr = len(dataloader_regr)

    # ae_loss_obs = LossTermObserver(n_epochs = epochs, n_iterations = n_iterations_ae)
    
    # loss_observer = CompositeLossTermObserver(
    #     n_epochs = epochs, 
    #     n_iterations = len(dataloader_regr),
    #     loss_names = ['Reconstruction Term', 'Regression Term'],
    # )


    ###--- Losses ---###
    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight= 1 - ete_regr_weight), 
        'Regression Term': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    
    ete_loss = Loss(CompositeLossTerm(**ete_loss_terms))
    reconstr_loss = Loss(loss_term = reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


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

    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = ae_loss_obs.losses)
    #loss_observer.plot_results()


    ###--- Test Loss ---###
    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    X_test_ul_hat = decoder(encoder((X_test_ul)))

    loss_reconst = reconstr_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)

    y_test_l_hat = regressor(encoder(X_test_l))

    loss_regr = regr_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)
    
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'n_layers_e': n_layers_e,
        'n_layers_d': n_layers_d,
        'encoder_lr': encoder_lr,
        'decoder_lr': decoder_lr,
        'regr_lr': regr_lr,
        'scheduler_gamma': scheduler_gamma,
        'ete_regr_weight': ete_regr_weight,
        'test_loss_reconst': loss_reconst.item(),
        'test_loss_regr': loss_regr.item(),
    }




@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="baseline_regr_cfg_sweep")
def train_baseline(cfg: DictConfig):

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size

    regr_lr = cfg.regr_lr
    scheduler_gamma = cfg.scheduler_gamma


    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.8)
    subsets = subset_factory.create_splits()

    regr_train_ds = subsets['train_labeled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    regressor = LinearRegr(latent_dim = input_dim)


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
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
    regr_test_ds = subsets['test_labeled']

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    y_test_l_hat = regressor(X_test_l)

    loss_regr = regr_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)
    
    return {
        'epochs': epochs,
        'batch_size': batch_size,
        'regr_lr': regr_lr,
        'scheduler_gamma': scheduler_gamma,
        'test_loss_regr': loss_regr.item(),
    }




"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":
    
    #experiment_name = 'joint_ae_regr_norm'
    #experiment_name = 'baseline_regr'
    experiment_name = 'ae_iso_test'

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
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
    )
    
    dataset = dataset_builder.build_dataset()



    ###--- Setup and calculate results ---###
    train_AE_iso_hydra()
    #train_VAE_iso_hydra()
    #train_joint_epoch_procedure()
    #train_baseline()

    