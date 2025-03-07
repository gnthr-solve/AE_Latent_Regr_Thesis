
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

from models.regressors import LinearRegr, DNNRegr
from models import AE, VAE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Loss, Weigh, WeightedCompositeLoss
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    LossTermVisitor
)

from helper_tools.setup import create_eval_metric
from helper_tools.ray_optim import RayTuneLossReporter

from ..config import ExperimentConfig


"""
Trainables - AE Linear Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_linear_joint_epoch(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):
    """
    Trainable for the autoencoder-linear model composition and joint-epoch training routine.

    Args:
    --------
        config: dict
            Config containing all hyperparameters choices for this trial.
        dataset: TensorDataset
        exp_cfg: ExperimentConfig
            Experiment config for access to optim_loss and eval_metrics.
    """
    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    n_layers = config['n_layers']
    activation = config['activation']

    encoder_lr = config['encoder_lr']
    decoder_lr = config['decoder_lr']
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']

    ete_regr_weight: float = config['ete_regr_weight']


    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size=0.9)
    train_subsets = subset_factory.retrieve(kind='train')
    
    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    ae_model_type = exp_cfg.model_params.get('AE_model_type', '')

    if ae_model_type == 'AE':
        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        
        ae_model = AE(encoder = encoder, decoder = decoder)

    elif ae_model_type == 'NVAE':
        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)

        ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    else:
        raise ValueError('Model not supported or specified')
    
    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    ae_loss = Loss(loss_term = reconstr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


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

            reconstr_component = reconstr_loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch).mean()
            regr_component = regr_loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch).mean()

            loss_ete_weighted = (1 - ete_regr_weight) * reconstr_component + ete_regr_weight * regr_component

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser.step()

        
        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = None
                
                torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
                torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


                checkpoint = Checkpoint.from_directory(tmp_dir)

                train.report({optim_loss: regr_component.item()}, checkpoint=checkpoint)
        else:
            train.report({optim_loss: regr_component.item()})


        scheduler.step()


    ###--- Evaluation ---###
    ae_model.eval()
    regressor.eval()
    test_subsets = subset_factory.retrieve(kind='test')

    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    ae_eval_metrics = {name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name.endswith('_reconstr')}
    regr_eval_metrics = {
        optim_loss: regr_loss_term,
        **{name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name not in ae_eval_metrics.keys()}
    }
    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = None
        
        torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
        torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


        checkpoint = Checkpoint.from_directory(tmp_dir)

        train.report(results.metrics, checkpoint=checkpoint)




"""
Trainables - AE Deep Joint Epoch
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_deep_joint_epoch(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):
    """
    Trainable for the autoencoder-deep model composition and joint-epoch training routine.
    
    Args:
    --------
        config: dict
            Config containing all hyperparameters choices for this trial.
        dataset: TensorDataset
        exp_cfg: ExperimentConfig
            Experiment config for access to optim_loss and eval_metrics.
    """
    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    #--- AE Params ---#
    n_layers = config['n_layers']
    activation_ae = config['activation']

    #--- DNNRegr Params ---#
    n_fixed_layers = config['n_fixed_layers']
    fixed_layer_size = config['fixed_layer_size']
    n_funnel_layers = config['n_funnel_layers']
    activation_regr = config['activation']

    #--- Optimisation ---#
    encoder_lr = config['encoder_lr']
    decoder_lr = config['decoder_lr']
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']

    ete_regr_weight: float = config['ete_regr_weight']


    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size=0.9)
    train_subsets = subset_factory.retrieve(kind='train')
    
    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    ae_model_type = exp_cfg.model_params.get('AE_model_type', '')

    if ae_model_type == 'AE':
        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation_ae)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation_ae)
        
        ae_model = AE(encoder = encoder, decoder = decoder)

    elif ae_model_type == 'NVAE':
        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation_ae)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation_ae)

        ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    else:
        raise ValueError('Model not supported or specified')
    
    regressor = DNNRegr(
        input_dim = latent_dim, 
        output_dim = 2,
        n_fixed_layers = n_fixed_layers,
        fixed_layer_size = fixed_layer_size,
        n_funnel_layers = n_funnel_layers,
        activation = activation_regr,
    )


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight = 1 - ete_regr_weight), 
        'Regression Term': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    ete_loss = Loss(CompositeLossTerm(ete_loss_terms))
    ae_loss = Loss(loss_term = reconstr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


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

            Z_batch, X_hat_batch = ae_model(X_batch)
            y_hat_batch = regressor(Z_batch)

            reconstr_component = reconstr_loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch).mean()
            regr_component = regr_loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch).mean()

            loss_ete_weighted = (1 - ete_regr_weight) * reconstr_component + ete_regr_weight * regr_component

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser.step()


        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = None
                
                torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
                torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


                checkpoint = Checkpoint.from_directory(tmp_dir)

                train.report({optim_loss: regr_component.item()}, checkpoint=checkpoint)
        else:
            train.report({optim_loss: regr_component.item()})



        scheduler.step()


    ###--- Evaluation ---###
    ae_model.eval()
    regressor.eval()
    test_subsets = subset_factory.retrieve(kind='test')

    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    ae_eval_metrics = {name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name.endswith('_reconstr')}
    regr_eval_metrics = {
        optim_loss: regr_loss_term,
        **{name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name not in ae_eval_metrics.keys()}
    }
    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = None
        
        torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
        torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


        checkpoint = Checkpoint.from_directory(tmp_dir)

        train.report(results.metrics, checkpoint=checkpoint)





"""
Trainables - AE Linear Joint Epoch EXPERIMENT
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_linear_joint_epoch_prime(config, dataset: TensorDataset, exp_cfg: ExperimentConfig):
    """
    Trainable for the autoencoder-linear model composition and joint-epoch training routine.
    For experiments.

    Args:
    --------
        config: dict
            Config containing all hyperparameters choices for this trial.
        dataset: TensorDataset
        exp_cfg: ExperimentConfig
            Experiment config for access to optim_loss and eval_metrics.
    """
    ###--- Experiment Meta ---###
    optim_loss = exp_cfg.optim_loss

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    n_layers = config['n_layers']
    activation = config['activation']

    encoder_lr = config['encoder_lr']
    decoder_lr = config['decoder_lr']
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']

    ete_regr_weight: float = config['ete_regr_weight']


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)

    reporter = RayTuneLossReporter(checkpoint_condition)
    

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size=0.9)
    train_subsets = subset_factory.retrieve(kind='train')
    
    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    ae_model_type = exp_cfg.model_params.get('AE_model_type', '')

    if ae_model_type == 'AE':
        encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
        
        ae_model = AE(encoder = encoder, decoder = decoder)

    elif ae_model_type == 'NVAE':
        encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)
        decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)

        ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    else:
        raise ValueError('Model not supported or specified')
    
    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    base_composite = CompositeLossTerm(
        loss_terms={
            'L2-norm_reconstr': reconstr_loss_term,
            optim_loss: regr_loss_term,
        },
        callbacks={optim_loss: [reporter.observe_loss]}
    )

    weights = {'L2-norm_reconstr': 1 - ete_regr_weight, optim_loss: ete_regr_weight}
    weighted_composite = WeightedCompositeLoss(base_composite, weights)
    
    ete_loss = Loss(weighted_composite)
    ae_loss = Loss(loss_term = reconstr_loss_term)
    

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

        
        #--- Model Checkpoints & Report ---#
        reporter.report(epoch = epoch, ae_model = ae_model, regressor = regressor)
        
        scheduler.step()


    ###--- Evaluation ---###
    ae_model.eval()
    regressor.eval()
    test_subsets = subset_factory.retrieve(kind='test')

    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    ae_eval_metrics = {name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name.endswith('_reconstr')}
    regr_eval_metrics = {
        optim_loss: regr_loss_term,
        **{name: create_eval_metric(name) for name in exp_cfg.eval_metrics if name not in ae_eval_metrics.keys()}
    }
    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = None
        
        torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
        torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


        checkpoint = Checkpoint.from_directory(tmp_dir)

        train.report(results.metrics, checkpoint=checkpoint)




