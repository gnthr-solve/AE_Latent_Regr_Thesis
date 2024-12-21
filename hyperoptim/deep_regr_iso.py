
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

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

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
    ReconstrLossVisitor, RegrLossVisitor,
)

from .config import ExperimentConfig

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def deep_regr(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):

    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    n_layers = config['n_layers']
    activation = config['activation']

    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']


    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    # Deep Regression
    regressor = DNNRegr(input_dim = input_dim, n_layers = n_layers, activation = activation)


    ###--- Losses ---###
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    regr_loss = Loss(loss_term = regr_loss_term)
    

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

        #--- Scheduler Step ---#
        scheduler.step()


    ###--- Test Loss ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'regressor': regressor},
    )

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso', loss_name = optim_loss)
    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        RegrLossVisitor(regr_loss_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_regr = results.metrics[eval_cfg.loss_name]
    
    train.report({optim_loss :loss_regr})
    
