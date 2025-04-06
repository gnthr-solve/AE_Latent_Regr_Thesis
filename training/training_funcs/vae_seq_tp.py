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
    VarEncoder,
    VarDecoder,
    SigmaGaussVarEncoder,
    SigmaGaussVarDecoder,
)

from models.regressors import LinearRegr, FunnelDNNRegr
from models import VAE, GaussVAE, GaussVAESigma

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.clt_callbacks import LossTrajectoryObserver
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver


from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    VAEOutputVisitor, RegrOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str

from visualisation.eval_plot_funcs import plot_3Dlatent_with_error, plot_3Dlatent_with_attribute
from visualisation.training_history_plots import plot_agg_training_losses



"""
Training Functions - VAE-Regressor Sequential
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def train_joint_seq_VAE():
    """
    Train VAE autoencoder first in isolation, then encoder-regressor composition.
    """
    ###--- Meta ---###
    epochs = 2
    batch_size = 100
    latent_dim = 3

    n_layers_e = 3
    n_layers_d = 3
    activation = 'Softplus'

    encoder_lr = 1e-3
    decoder_lr = 1e-3
    regr_lr = 1e-2
    scheduler_gamma = 0.9

    ete_regr_weight = 0.95

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

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    ae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    dataset_size_ete = len(regr_train_ds)
    observer_callback = LossTrajectoryObserver()


    ###--- Loss Terms ---###
    #--- VAE ---#
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTerm(
        loss_terms = vae_loss_terms,
        callbacks = {name: [observer_callback] for name in vae_loss_terms.keys()},
    )

    ae_loss = Loss(vae_loss_term)

    #--- Reconstruction for testing ---#
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))

    #--- Regression ---#
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))
    
    #--- Composite ETE ---#
    ete_loss_terms = {
        'Reconstruction Term': Weigh(vae_loss_term, weight = 1 - ete_regr_weight), 
        'Regression Term': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    ete_loss = Loss(
        CompositeLossTerm(
            loss_terms = ete_loss_terms,
            callbacks = {name: [observer_callback] for name in ete_loss_terms.keys()},
        )
    )
    

    ###--- Optimizer & Scheduler ---###
    optimiser_ae = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
    ])

    optimiser_regr = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 5e-3},
    ])

    scheduler_ae = ExponentialLR(optimiser_ae, gamma = 0.5)
    scheduler_regr = ExponentialLR(optimiser_regr, gamma = 0.5)


    ###--- Training Loop AE---###
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_ae.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = ae_model(X_batch)

            loss_ae = ae_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )


            #--- Backward Pass ---#
            loss_ae.backward()

            optimiser_ae.step()

            #--- Observer Call ---#
            #vae_model_obs(epoch = epoch, iter_idx = iter_idx, model = vae_model)
            #vae_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_vae)

        scheduler_ae.step()


    ###--- Training Loop Regr ---###
    for epoch in pbar:
        
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_regr.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = ae_model(X_batch)
            y_hat_batch = regressor(Z_batch)
            
            loss_ete_weighted = ete_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
                y_batch = y_batch, 
                y_hat_batch = y_hat_batch,
            )
        
            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser_regr.step()

            #--- Observer Call ---#
            #regr_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_regr)


        scheduler_regr.step()


    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = vae_loss_obs.losses)
    #plot_loss_tensor(observed_losses = regr_loss_obs.losses)

    #vae_model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))
    

    ###--- Test Loss ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')

    visitors = [
        VAEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(reconstr_loss_term, loss_name = 'L2_norm', eval_cfg = eval_cfg_reconstr),

        VAEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(regr_loss_term, loss_name = 'Huber', eval_cfg = eval_cfg_comp),
        LatentPlotVisitor(eval_cfg = eval_cfg_comp)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[eval_cfg_reconstr.loss_name]
    loss_regr = results.metrics[eval_cfg_comp.loss_name]

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconstr}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )



