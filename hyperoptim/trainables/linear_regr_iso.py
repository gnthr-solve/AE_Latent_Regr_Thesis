
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
    ReconstrLossVisitor, RegrLossVisitor,
)

from ..config import ExperimentConfig

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def linear_regr(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):

    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']
    

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    # Linear Regression
    regressor = LinearRegr(latent_dim = input_dim)


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(loss_term = regr_loss_term)
    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


    ###--- Training Loop Joint---###
    for epoch in range(epochs):
        
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
        
        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                #context = train.get_context()
                
                torch.save(
                    regressor.state_dict(),
                    os.path.join(temp_checkpoint_dir, f"regressor.pt"),
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                train.report({optim_loss: loss_regr.item()}, checkpoint=checkpoint)
        else:
            train.report({optim_loss: loss_regr.item()})

        scheduler.step()

    
    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    
    regr_test_ds = test_subsets['labelled']

    test_indices = regr_test_ds.indices
    X_test_l = dataset.X_data[test_indices]
    y_test_l = dataset.y_data[test_indices]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():
        y_test_l_hat = regressor(X_test_l)

        loss_regr = regr_loss_test(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    train.report({optim_loss :loss_regr.item()})
