import os
import tempfile
import torch
import ray
import logging

from ray import train, tune
from ray.train import Checkpoint

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path

from data_utils import TensorDataset, SplitSubsetFactory

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, ProductRegr, DNNRegr
from models import AE, VAE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    CompositeLossTerm,
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

from helper_tools.setup import create_eval_metric

from ..config import ExperimentConfig

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def VAE_iso(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):
    
    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    n_layers_e = config['n_layers_e']
    n_layers_d = config['n_layers_d']
    activation = config['activation']
    beta = config['beta']

    ae_lr = config['ae_lr']
    scheduler_gamma = config['scheduler_gamma']

    
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
    
    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}

    ae_loss = Loss(CompositeLossTerm(loss_terms))
    eval_ae_loss_term = AEAdapter(LpNorm(p = 2))


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam(ae_model.parameters(), lr = ae_lr)
    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


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

        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = None
                #context = train.get_context()
                
                torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))


                checkpoint = Checkpoint.from_directory(tmp_dir)

                #NOTE: This reporting needs to be adjusted because the ETE loss is not the same as the regression loss
                train.report({optim_loss: loss_ae.item()}, checkpoint=checkpoint)
        else:
            train.report({optim_loss: loss_ae.item()})

        scheduler.step()


    ###--- Test Loss ---###
    ae_model.eval()
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'joint': test_dataset},
        models = {'AE_model': ae_model},
    )

    eval_metrics = {optim_loss: eval_ae_loss_term, **{name: create_eval_metric(name) for name in exp_cfg.eval_metrics}}
    eval_cfg = EvalConfig(data_key = 'joint', output_name = 'ae_iso', mode = 'iso')

    ae_output_visitor = VAEOutputVisitor(eval_cfg = eval_cfg)
    
    visitors = [
        ae_output_visitor,
        LossTermVisitor(loss_terms = eval_metrics, eval_cfg = eval_cfg)
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results

    train.report(results.metrics)

