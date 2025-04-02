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

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver
from observers.latent_visualiser import LatentSpaceVisualiser

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str

from visualisation.eval_plot_funcs import plot_3Dlatent_with_error, plot_3Dlatent_with_attribute
from visualisation.training_history_plots import plot_agg_training_losses



"""
Training Functions - AE Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_joint_epoch_procedure():
    """
    Joint epoch training
    1. AE/NVAE on reconstruction loss
    2. AE-Regressor composition on reconstruction and regression loss.
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

    ete_regr_weight = 0.95

    dataset_kind = 'key'
    normaliser_kind = 'min_max'
    filter_condition = filter_by_machine('M_A')


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        #normaliser = normaliser,
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


    ###--- Models ---###
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
    observer_callback = LossTrajectoryObserver()


    ###--- Losses ---###
    ae_loss_name = 'L2_reconstr'
    regr_loss_name = 'huber_regr'
    loss_terms = {
        ae_loss_name: AEAdapter(LpNorm(p = 2)), 
        regr_loss_name: RegrAdapter(Huber(delta = 1)),
    }

    ete_clt = CompositeLossTerm(
        loss_terms = loss_terms,
        callbacks = {name: [observer_callback] for name in loss_terms.keys()}
    )

    ete_clt = WeightedCompositeLoss(composite_lt=ete_clt, weights={ae_loss_name: 1 - ete_regr_weight, regr_loss_name: ete_regr_weight})

    ete_loss = Loss(ete_clt)
    ae_loss = Loss(loss_terms[ae_loss_name])
    

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
            
            loss_ete_weighted = ete_loss(
                X_batch = X_batch, 
                X_hat_batch = X_hat_batch,
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
        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(loss_terms[ae_loss_name], ae_loss_name, eval_cfg = eval_cfg_reconstr),

        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(loss_terms[regr_loss_name], regr_loss_name, eval_cfg = eval_cfg_comp),
        #LatentPlotVisitor(eval_cfg = eval_cfg_comp, loss_name = regr_loss_name),
        LossStatisticsVisitor(loss_name = regr_loss_name, eval_cfg = eval_cfg_comp)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[ae_loss_name]
    loss_regr = results.metrics[regr_loss_name]

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconstr}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n\n"

        f"All Metrics:\n"
        f"---------------------------------------------------------------\n"
        f"{dict_str(results.metrics)}\n"
    )

    # if latent_dim == 3:
    #     title = f'NVAE Normalised (epochs = {epochs})'
    #     plot_3Dlatent_with_error(
    #         latent_tensor = Z_batch_ul,
    #         loss_tensor = loss_reconst,
    #         title = title
    #     )
    #     title = f'NVAE Normalised Regr(epochs = {epochs})'
    #     plot_3Dlatent_with_error(
    #         latent_tensor = Z_batch_l,
    #         loss_tensor = loss_regr,
    #         title = title
    #     )




"""
Training Functions - AE Joint Epoch - Loss tests
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_regr_loss_tests():
    """
    Joint epoch training with additional weighted loss compositions.
    1. AE/NVAE on 
    AE loss = weighted sum of (reconstruction loss, topological loss | clustering loss).
    2. AE-Regressor composition on 
    ETE loss = weighted sum of (AE loss,  regression loss).
    """
    ###--- Meta ---###
    epochs = 5
    batch_size = 100
    latent_dim = 2

    ae_model_type = 'NVAE'
    n_layers_e = 5
    n_layers_d = 5
    activation = 'Softplus'

    encoder_lr = 1e-3
    decoder_lr = 1e-3
    regr_lr = 1e-2
    scheduler_gamma = 0.9

    ae_base_weight = 0.5
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
    observer_callback = LossTrajectoryObserver()
    latent_visualiser = LatentSpaceVisualiser(
        output_dir = './results/AE_regr_loss_tests/latent_frames',
        iter_label = 'iter'
    )


    ###--- Losses ---###
    loss_terms = {
        'L2': AEAdapter(LpNorm(p = 2)),
        'topo': Topological(p = 2),
        'kmeans': KMeansLoss(n_clusters = 5, latent_dim = latent_dim),
        'Huber': RegrAdapter(Huber(delta = 1)),
    }

    ae_loss_base_name = 'L2'
    ae_loss_extra_name = 'topo'
    #ae_loss_extra_name = 'kmeans'
    regr_loss_name = 'Huber'

    ae_clt = CompositeLossTerm(
        loss_terms = {ae_loss_base_name: loss_terms[ae_loss_base_name], ae_loss_extra_name: loss_terms[ae_loss_extra_name]}
    )

    ae_clt = WeightedCompositeLoss(
        composite_lt = ae_clt, 
        weights={ae_loss_base_name: ae_base_weight, ae_loss_extra_name: 1 - ae_base_weight}
    )
    ae_clt.add_callback(name = 'ALL', callback = observer_callback)

    ete_loss_terms = {
        'ae_loss': ae_clt,
        'regr_loss': loss_terms[regr_loss_name],
    }

    ete_clt = CompositeLossTerm(loss_terms = ete_loss_terms)
    ete_clt = WeightedCompositeLoss(
        composite_lt=ete_clt, 
        weights={'ae_loss': 1 - ete_regr_weight, regr_loss_name: ete_regr_weight}
    )

    ete_loss = Loss(ete_clt)
    ae_iso_loss = Loss(ae_clt)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 40
    epoch_modulo = len(dataloader_ae) * epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)

    ###--- Training Procedure ---###
    for epoch in range(epochs):
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)

            loss_ae = ae_iso_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch, 
                Z_batch = Z_batch,
            )
            
            if checkpoint_condition(iter_idx):
                latent_visualiser(latent_vectors = Z_batch, iteration = iter_idx, loss = loss_ae)

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

            loss_ete_weighted = ete_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
                Z_batch = Z_batch,
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

    latent_visualiser.finalize('./results/AE_regr_loss_tests/latent_evolution.mp4')

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
        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitorS(loss_terms[ae_loss_base_name], loss_name= ae_loss_base_name, eval_cfg = eval_cfg_reconstr),

        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(loss_terms[regr_loss_name], loss_name = regr_loss_name, eval_cfg = eval_cfg_comp),
        #LatentPlotVisitor(eval_cfg = eval_cfg_comp, loss_name = regr_loss_name)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[ae_loss_base_name]
    loss_regr = results.metrics[regr_loss_name]

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


