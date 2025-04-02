###--- External Library Imports ---###
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm


###--- Custom Imports ---###
from data_utils import DatasetBuilder, SplitSubsetFactory, retrieve_metadata
from data_utils.info import time_col, exclude_columns
from data_utils.data_filters import filter_by_machine

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
    SigmaGaussVarEncoder,
    SigmaGaussVarDecoder,
)

from models.regressors import LinearRegr, FunnelDNNRegr
from models import AE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    KMeansLoss,
)

from loss.clt_callbacks import LossTrajectoryObserver
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver
from observers.latent_visualiser import LatentSpaceVisualiser

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, RegrOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str

from visualisation.eval_plot_funcs import plot_3Dlatent_with_error, plot_3Dlatent_with_attribute
from visualisation.training_history_plots import plot_agg_training_losses



"""
Training Functions - AE-Regressor Sequential
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def train_joint_seq_AE():
    """
    Train autoencoder first in isolation, then encoder-regressor composition.
    """
    ###--- Meta ---###
    epochs = 4
    batch_size = 50
    latent_dim = 10

    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)
    
    dataset_builder = DatasetBuilder(
        kind = 'max',
        normaliser = normaliser,
        #exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()
    
    
    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 5)

    decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 5)

    regressor = LinearRegr(latent_dim = latent_dim)

    regr_model = EnRegrComposite(encoder = encoder, regressor = regressor)
    ae_model = AE(encoder = encoder, decoder = decoder)


    ###--- Observation Test Setup ---###
    n_iterations_ae = len(dataloader_ae)
    n_iterations_regr = len(dataloader_regr)

    #ae_model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations_ae, model = ae_model)
    #ae_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_ae)

    regr_model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations_regr, model = regr_model)
    #regr_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_regr)


    ###--- Losses ---###
    #reconstr_loss = Loss(LpNorm(p = 2))
    reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))
    regr_loss = Loss(RegrAdapter(Huber(delta = 1)))


    ###--- Optimizer & Scheduler ---###
    optimizer_ae = Adam(ae_model.parameters(), lr = 1e-2)
    scheduler_ae = ExponentialLR(optimizer_ae, gamma = 0.9)

    optimizer_regr = Adam(regr_model.parameters(), lr = 1e-2)
    scheduler_regr = ExponentialLR(optimizer_regr, gamma = 0.9)


    ###--- Training Loop AE---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_ae.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            optimizer_ae.step()

            #--- Observer Call ---#
            #ae_model_obs(epoch = epoch, iter_idx = iter_idx, model = ae_model)
            #ae_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_reconst)


        scheduler_ae.step()



    ###--- Training Loop Regr ---###
    for epoch in pbar:
        
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_regr.zero_grad()
            
            y_hat_batch = regr_model(X_batch)

            loss_regr = regr_loss(y_batch = y_batch, y_hat_batch = y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            optimizer_regr.step()

            #--- Observer Call ---#
            regr_model_obs(epoch = epoch, iter_idx = iter_idx, model = regr_model)
            #regr_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_regr)

        scheduler_regr.step()


    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = ae_loss_obs.losses)
    #plot_loss_tensor(observed_losses = regr_loss_obs.losses)

    #ae_model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))
    regr_model_obs.plot_child_param_development(child_name = 'regressor', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    ae_test_ds = test_subsets['unlabelled']
    regr_test_ds = test_subsets['labelled']

    X_test_ae = dataset.X_data[ae_test_ds.indices]
    X_test_regr = dataset.X_data[regr_test_ds.indices]
    y_test_regr =dataset.y_data[regr_test_ds.indices]
    X_test_ae = X_test_ae[:, 1:]
    X_test_regr = X_test_regr[:, 1:]
    y_test_regr = y_test_regr[:, 1:]

    Z_batch_test, X_test_hat = ae_model(X_test_ae)
    y_test_hat = regr_model(X_test_regr)

    loss_reconst = reconstr_loss(X_batch = X_test_ae, X_hat_batch = X_test_hat)
    loss_regr = regr_loss(y_batch = y_test_regr, y_hat_batch = y_test_hat)

    print(
        f"After {epochs} epochs\n"
        f"------------------------------------------------\n"
        f"AE iterations per Epoch: \n{len(dataloader_ae)}\n"
        f"Regr iterations per Epoch: \n{len(dataloader_regr)}\n"
        f"------------------------------------------------\n"
        f"Avg. reconstruction loss on testing subset: {loss_reconst}\n"
        f"Avg. regression loss on testing subset: {loss_regr}\n"
        f"------------------------------------------------\n"
    )




def train_seq_AE():
    """
    Train autoencoder first in isolation, then autoencoder-regressor composition (including decoder).
    """
    ###--- Meta ---###
    epochs = 2
    batch_size = 50
    latent_dim = 3

    ae_model_type = 'NVAE'
    n_layers_e = 5
    n_layers_d = 5
    activation = 'Softplus'

    encoder_lr = 1e-3
    decoder_lr = 1e-3
    regr_lr = 1e-2
    scheduler_gamma = 0.9

    dataset_kind = 'key'
    normaliser_kind = 'min_max'
    filter_condition = filter_by_machine('M_A')


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns,
        #filter_condition = filter_condition,
    )
    
    dataset = dataset_builder.build_dataset()
    print(f'Dataset size: {len(dataset)}')
    
    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ####--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    if ae_model_type == 'AE':
        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e, activation = activation)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d, activation = activation)
        
        ae_model = AE(encoder = encoder, decoder = decoder)

    elif ae_model_type == 'NVAE':
        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

        ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    else:
        raise ValueError('Model not supported or specified')
    
    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    n_iterations_ae = len(dataloader_ae)
    n_iterations_regr = len(dataloader_regr)

    #ae_model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations_ae, model = ae_model)
    #ae_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_ae)

    #regr_model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations_regr, model = regressor)
    #regr_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_regr)


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    regr_loss_term = RegrAdapter(Huber(delta = 1))

    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimizer_ae = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
    ])
    scheduler_ae = ExponentialLR(optimizer_ae, gamma = scheduler_gamma)

    optimizer_regr = Adam(regressor.parameters(), lr = regr_lr)
    scheduler_regr = ExponentialLR(optimizer_regr, gamma = scheduler_gamma)


    ###--- Training Loop AE---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_ae.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            optimizer_ae.step()

            #--- Observer Call ---#
            #ae_model_obs(epoch = epoch, iter_idx = iter_idx, model = ae_model)
            #ae_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_reconst)


        scheduler_ae.step()


    ###--- Training Loop Regr ---###
    for epoch in pbar:
        
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_regr.zero_grad()
            
            Z_batch, _ = ae_model(X_batch)
            y_hat_batch = regressor(Z_batch)

            loss_regr = regr_loss(y_batch = y_batch, y_hat_batch = y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            optimizer_regr.step()

            #--- Observer Call ---#
            #regressor_obs(epoch = epoch, iter_idx = iter_idx, model = regressor)
            #regr_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_regr)

        scheduler_regr.step()


    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = ae_loss_obs.losses)
    #plot_loss_tensor(observed_losses = regr_loss_obs.losses)

    #ae_model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))
    #regressor_obs.plot_child_param_development(child_name = 'regressor', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Evaluation ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')

    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(reconstr_loss_term, 'L2_norm', eval_cfg = eval_cfg_reconstr),

        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(regr_loss_term, 'Huber', eval_cfg = eval_cfg_comp),
        LatentPlotVisitor(eval_cfg = eval_cfg_comp, loss_name ='Huber')
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics['L2_norm']
    loss_regr = results.metrics['Huber']

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconstr}\n\n"
        
        f"Regression:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )


