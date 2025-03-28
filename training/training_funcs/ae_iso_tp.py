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

from ..procedure_iso import AEIsoTrainingProcedure

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LatentPlotVisitor, LatentDistributionVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str
from visualisation.general_plot_funcs import (
    plot_loss_tensor,
    plot_agg_training_losses,
    plot_3Dlatent_with_error, 
    plot_3Dlatent_with_attribute,
)



"""
Training Functions - AE/NVAE Iso
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_iso_training_procedure():
    """
    Train deterministic or NVAE Autoencoder in isolation.
    """
    ###--- Meta ---###
    epochs = 3
    batch_size = 50
    latent_dim = 3
    lr = 1e-3
    gamma = 0.9
    print(f"Learning rates: {[lr * gamma**epoch for epoch in range(epochs)]}")

    dataset_kind = 'key'

    n_layers_e = 5
    n_layers_d = 5
    activation = 'PReLU'

    model_kind = 'stochastic'
    model_type = 'NVAE'

    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        #exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()
    
    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Model ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    if model_kind == 'deterministic':

        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e, activation = activation)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d, activation = activation)

    elif model_kind == 'stochastic':

        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    if model_type == 'AE':
        model = AE(encoder = encoder, decoder = decoder)
    elif model_type == 'NVAE':
        model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    elif model_type == 'VAE':
        model = GaussVAE(encoder = encoder, decoder = decoder)

    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    #latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    
    
    ###--- Loss ---###
    if isinstance(model, VAE):

        ll_term = Weigh(GaussianDiagLL(), weight = -1)
        kld_term = GaussianAnaKLDiv()
        
        loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}

    elif isinstance(model, AE):

        reconstr_term = AEAdapter(LpNorm(p = 2))
        loss_terms = {'Reconstruction': reconstr_term}

    
    train_loss = Loss(CompositeLossTerm(loss_terms = loss_terms))
    test_reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = lr)
    scheduler = ExponentialLR(optimizer, gamma = gamma)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = train_loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )
    
    training_procedure()

    ###--- Test Loss ---###
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'joint': test_dataset},
        models = {'AE_model': model},
    )

    eval_cfg = EvalConfig(data_key = 'joint', output_name = 'ae_iso', mode = 'iso')

    ae_output_visitor = VAEOutputVisitor(eval_cfg = eval_cfg) if model_type == 'VAE' else AEOutputVisitor(eval_cfg = eval_cfg)
    dist_vis_visitor = LatentPlotVisitor(eval_cfg = eval_cfg) if latent_dim == 3 else LatentDistributionVisitor(eval_cfg = eval_cfg)
    visitors = [
        ae_output_visitor,
        LossTermVisitorS(test_reconstr_term, loss_name = 'rel_L2_loss', eval_cfg = eval_cfg),
        dist_vis_visitor,
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    print(
        f"After {epochs} epochs with {n_iterations} iterations each\n"
        f"Avg. Loss on testing subset: {results.metrics[eval_cfg.loss_name]}\n"
    )

    # if latent_dim == 3:
    #     title = f'NVAE Normalised MinMax (epochs = {epochs})'
    #     plot_3Dlatent_with_error(
    #         latent_tensor = Z_batch_hat,
    #         loss_tensor = loss_reconst,
    #         title = title
    #     )

