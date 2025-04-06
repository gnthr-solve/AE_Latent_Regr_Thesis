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
from models import AE, VAE, GaussVAE, GaussVAESigma, EnRegrComposite
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
from loss.topology_term import Topological
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver
from observers.training_observer import TrainingObserver
from observers.observations_converter import TrainingObsConverter

from ..procedure_iso import AEIsoTrainingProcedure

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str
from visualisation.eval_plot_funcs import plot_3Dlatent_with_error, plot_3Dlatent_with_attribute
from visualisation.training_history_plots import plot_agg_training_losses, plot_2Dlatent_by_epoch



"""
Training Functions - VAE Iso
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def VAE_iso():

    ###--- Meta ---###
    epochs = 2
    batch_size = 100
    latent_dim = 3

    n_layers_e = 3
    n_layers_d = 3
    activation = 'Softplus'
    beta = 1

    ae_lr = 1e-3
    scheduler_gamma = 0.9

    dataset_kind = 'key'
    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)
    
    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns
    )
    
    dataset = dataset_builder.build_dataset()

    
    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Model ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    ae_model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)
    kld_term = Weigh(GaussianAnaKLDiv(), weight = beta)
    #kld_term = Weigh(GaussianMCKLDiv(receives_logvar = use_logvar), weight = beta)
    
    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}

    ae_loss = Loss(CompositeLossTerm(loss_terms))
    
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam(ae_model.parameters(), lr = ae_lr)
    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    for epoch in range(epochs):
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = ae_model(X_batch)

            loss_ae = ae_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )

            #--- Backward Pass ---#
            loss_ae.backward()
            optimiser.step()

        scheduler.step()


    ###--- Test Loss ---###
    ae_model.eval()
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'joint': test_dataset},
        models = {'AE_model': ae_model},
    )

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    
    visitors = [
        VAEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(reconstr_loss_term, loss_name = 'L2_norm', eval_cfg = eval_cfg_reconstr),
        LatentPlotVisitor(eval_cfg = eval_cfg_reconstr)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics['L2_norm']

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconstr}\n\n"
    )



"""
Training Functions - VAE Iso
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def VAE_iso_observer_testing():

    ###--- Meta ---###
    epochs = 2
    batch_size = 100
    latent_dim = 2

    n_layers_e = 3
    n_layers_d = 3
    activation = 'Softplus'
    beta = 1

    ae_lr = 1e-3
    scheduler_gamma = 0.9

    dataset_kind = 'key'
    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)
    
    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns
    )
    
    dataset = dataset_builder.build_dataset()

    
    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Model ---###
    input_dim = dataset.X_dim - 1

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    ae_model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)
    observer = TrainingObserver(
        n_epochs = epochs,
        iterations_per_epoch = n_iterations,
    )


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)
    kld_term = Weigh(GaussianAnaKLDiv(), weight = beta)
    #kld_term = Weigh(GaussianMCKLDiv(receives_logvar = use_logvar), weight = beta)
    
    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}

    ae_loss = Loss(CompositeLossTerm(loss_terms))
    
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam(ae_model.parameters(), lr = ae_lr)
    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    for epoch in range(epochs):
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = ae_model(X_batch)

            loss_ae = ae_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )

            observer(latent_vars = Z_batch, loss = loss_ae)

            #--- Backward Pass ---#
            loss_ae.backward()
            optimiser.step()

        scheduler.step()


    ###--- Handle Observations ---###
    observations = observer.get_tensor(name = 'latent_vars')

    converter = TrainingObsConverter(observations)
    batch_agg_observations = converter.to_dict_by_epoch_batch_split(n_epochs = epochs, batch_size = batch_size)

    plot_2Dlatent_by_epoch(latent_observations = batch_agg_observations)


    ###--- Test Loss ---###
    ae_model.eval()
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'joint': test_dataset},
        models = {'AE_model': ae_model},
    )

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    
    visitors = [
        VAEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(reconstr_loss_term, loss_name = 'L2_norm', eval_cfg = eval_cfg_reconstr),
        #LatentPlotVisitor(eval_cfg = eval_cfg_reconstr)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics['L2_norm']

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconstr}\n\n"
    )




