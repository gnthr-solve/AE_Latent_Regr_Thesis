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
from models import VAE, GaussVAE, GaussVAESigma, EnRegrComposite

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

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver
from observers.latent_visualiser import LatentSpaceVisualiser

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
Training Functions - VAE Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def train_joint_epoch_wise_VAE_recon():
    """
    Joint epoch training
    1. VAE on ELBO
    2. VAE-Regressor on reconstruction and regression loss.
    """
    ###--- Meta ---###
    epochs = 2
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
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    #print(len(dataloader_ae), len(dataloader_regr))


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    n_iterations_vae = len(dataloader_ae)
    n_iterations_regr = len(dataloader_regr)
    dataset_size_ete = len(regr_train_ds)

    #vae_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_vae)
    

    ###--- Loss Terms ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    #kld_term = GaussianAnaKLDiv()
    kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_elbo_term = CompositeLossTerm(**vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight=0.1), 
        'Regression Term': Weigh(regr_loss_term, weight = 0.9),
    }

    ###--- Losses ---###
    vae_loss = Loss(vae_elbo_term)
    ete_loss = Loss(CompositeLossTerm(loss_terms = ete_loss_terms))
    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser_vae = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
    ])

    optimiser_ete = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 5e-3},
    ])

    scheduler_vae = ExponentialLR(optimiser_vae, gamma = 0.5)
    scheduler_ete = ExponentialLR(optimiser_ete, gamma = 0.5)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_vae.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)

            loss_vae = vae_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )


            #--- Backward Pass ---#
            loss_vae.backward()

            optimiser_vae.step()

            #--- Observer Call ---#
            #vae_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_vae)


        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_ete.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)
            X_hat_batch, _ = genm_dist_params.unbind(dim = -1)

            y_hat_batch = regressor(Z_batch)


            loss_ete_weighted = ete_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser_ete.step()


        scheduler_vae.step()
        scheduler_ete.step()


    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = vae_loss_obs.losses)
    #loss_observer.plot_agg_results()


    ###--- Test Loss ---###
    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    #--- Select Test-Data ---#
    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    #--- Apply VAE to labelled and unlabelled data ---#
    Z_batch_l, infrm_dist_params_l, genm_dist_params_l = vae_model(X_test_l)
    Z_batch_ul, infrm_dist_params_ul, genm_dist_params_ul = vae_model(X_test_ul)

    #--- Reconstruction  ---#
    mu_r, _ = genm_dist_params_ul.unbind(dim = -1)
    X_test_ul_hat = mu_r

    loss_reconst = reconstr_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)


    y_test_l_hat = regressor(Z_batch_l)

    loss_regr = regr_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconst}\n\n"
        
        f"Regression Trained End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )




def VAE_joint_epoch_procedure():
    """
    Joint epoch training
    1. VAE on ELBO
    2. VAE-Regressor composition on ELBO and regression loss.
    """
    ###--- Meta ---###
    epochs = 5
    batch_size = 100
    latent_dim = 3

    n_layers_e = 3
    n_layers_d = 3
    activation = 'Softplus'
    use_logvar = False

    encoder_lr = 1e-3
    decoder_lr = 1e-3
    regr_lr = 1e-2
    scheduler_gamma = 0.9

    ete_regr_weight = 0.95

    dataset_kind = 'key'
    #normaliser_kind = 'min_max'
    normaliser_kind = None
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

    if use_logvar:
        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

        ae_model = GaussVAE(encoder = encoder, decoder = decoder)

    else:
        encoder = SigmaGaussVarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e, activation = activation)
        decoder = SigmaGaussVarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d, activation = activation)

        ae_model = GaussVAESigma(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    dataset_size_ete = len(regr_train_ds)
    observer_callback = LossTrajectoryObserver()


    ###--- Loss Terms ---###
    #--- VAE ---#
    ll_term = GaussianDiagLL(receives_logvar = use_logvar)
    kld_term = GaussianAnaKLDiv(receives_logvar = use_logvar)
    #kld_term = GaussianMCKLDiv(receives_logvar = use_logvar)

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTerm(
        loss_terms = vae_loss_terms,
        callbacks = {name: [observer_callback] for name in vae_loss_terms.keys()},
    )
    vae_loss_term.apply_decorator(target_name = 'Log-Likelihood', decorator_cls = Weigh, weight = -1)

    ae_loss = Loss(vae_loss_term)

    #--- Reconstruction for testing ---#
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))

    #--- Regression ---#
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))
    
    #--- Composite ETE ---#
    ete_loss_terms = {
        'Reconstr': Weigh(vae_loss_term, weight = 1 - ete_regr_weight), 
        'Regr': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    ete_loss = Loss(
        CompositeLossTerm(
            loss_terms = ete_loss_terms,
            callbacks = {name: [observer_callback] for name in ete_loss_terms.keys()},
        )
    )
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Procedure ---###
    for epoch in range(epochs):
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
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


        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
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

            optimiser.step()

        
        scheduler.step()


    ###--- Plot Observations ---###
    observed_losses_dict = observer_callback.get_history(concat = True)
    plot_agg_training_losses(
        training_losses = observed_losses_dict,
        epochs = epochs,
    )


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
        LossTermVisitorS(reconstr_loss_term, loss_name = 'L2_error_reconstr', eval_cfg = eval_cfg_reconstr),

        VAEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(regr_loss_term, loss_name = 'L2_error_regr', eval_cfg = eval_cfg_comp),
        LatentPlotVisitor(eval_cfg = eval_cfg_comp, loss_name = 'L2_error_regr')
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics['L2_error_reconstr']
    loss_regr = results.metrics['L2_error_regr']

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


