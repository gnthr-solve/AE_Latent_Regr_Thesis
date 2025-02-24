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
from data_utils.info import time_col
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
)

from loss.clt_callbacks import LossTrajectoryObserver
from loss.topology_term import Topological
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

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
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_iso_training_procedure():
    
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
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def train_joint_seq_AE():

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
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def train_joint_seq_VAE():

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
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def train_seq_AE():

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
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def train_joint_epoch_wise_VAE_recon():

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
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def AE_joint_epoch_procedure():

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
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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
        #LatentPlotVisitor(eval_cfg = eval_cfg_comp, loss_name ='Huber'),
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




def VAE_joint_epoch_procedure():

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
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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




def train_linear_regr():

    ###--- Meta ---###
    epochs = 100
    batch_size = 20

    regr_lr = 1e-2
    scheduler_gamma = 0.99

    dataset_kind = 'key'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]

    observe_loss_dev = False
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

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    # Linear Regression
    regressor = LinearRegr(latent_dim = input_dim)

    
    ###--- Observation Test Setup ---###
    n_iterations_regr = len(dataloader_regr)
    dataset_size = len(regr_train_ds)


    ###--- Losses ---###
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    if observe_loss_dev:

        loss_obs = LossTermObserver(
            n_epochs = epochs,
            dataset_size= dataset_size,
            batch_size= batch_size,
            name = 'Regr Loss',
            aggregated = True,
        )

        regr_loss = Loss(Observe(observer = loss_obs, loss_term = regr_loss_term))
    
    else:
        regr_loss = Loss(loss_term = regr_loss_term)

    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
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


        scheduler.step()

    
    ###--- Plot Observations ---###
    if observe_loss_dev:
        plot_loss_tensor(observed_losses = loss_obs.losses)


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

    print(
        f"Regression Baseline:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )

    ###--- Experiment ---###
    import matplotlib.pyplot as plt

    #print(f'Regr weight shape: {regressor.regr_map.weight.shape}')
    weights_1, weights_2 = regressor.regr_map.weight.detach().unbind(dim = 0)
    bias_1, bias_2 = regressor.regr_map.bias.detach().unbind(dim = 0)

    col_indices = np.arange(1, dataset.X_dim)
    X_col_labels = dataset.alignm.retrieve_col_labels(indices = col_indices)
    y_col_labels = dataset.alignm.retrieve_col_labels(indices = [1,2], from_X = False)

    weight_df_1 = pd.DataFrame({'Feature': X_col_labels, 'Weight': weights_1.numpy()})
    weight_df_2 = pd.DataFrame({'Feature': X_col_labels, 'Weight': weights_2.numpy()})

    for label, weight_df, bias in zip(y_col_labels, [weight_df_1, weight_df_2], [bias_1, bias_2]):

        weight_df['Absolute Weight'] = weight_df['Weight'].abs()
        weight_df = weight_df.sort_values(by = 'Absolute Weight', ascending = False)
        print(
            f'{label}:\n'
            f'-------------------------------------\n'
            f'weights:\n{weight_df}\n'
            f'bias:\n{bias}\n'
            f'-------------------------------------\n'
        )

        n_top_features = 20
        top_features_df = weight_df.head(n_top_features)

        # Plot the feature importance for the top 20 features
        plt.figure(figsize=(14, 6))
        plt.tight_layout()
        plt.barh(top_features_df['Feature'], top_features_df['Weight'], color='steelblue')
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {n_top_features} Feature Weights in Linear Model for {label}')
        plt.gca().invert_yaxis()  # Most important features on top
        plt.show()




def train_deep_regr():

    ###--- Meta ---###
    epochs = 5
    batch_size = 30

    n_layers = 5
    activation = 'Softplus'

    regr_lr = 1e-2
    scheduler_gamma = 0.9

    dataset_kind = 'key'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")


    # Deep Regression
    regressor = FunnelDNNRegr(input_dim = input_dim, n_layers = n_layers, activation = activation)

    ###--- Observation Test Setup ---###
    n_iterations_regr = len(dataloader_regr)
    dataset_size = len(regr_train_ds)

    loss_obs = LossTermObserver(
        n_epochs = epochs,
        dataset_size= dataset_size,
        batch_size= batch_size,
        name = 'Regr Loss',
        aggregated = True,
    )


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(Observe(observer = loss_obs, loss_term = regr_loss_term))
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
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


        scheduler.step()

    
    ###--- Plot Observations ---###
    plot_loss_tensor(observed_losses = loss_obs.losses)


    ###--- Test Loss ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'regressor': regressor},
    )

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso', loss_name = 'Huber')
    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        LossTermVisitorS(regr_loss_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_regr = results.metrics[eval_cfg.loss_name]
    print(
        f"Regression Baseline:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )




def AE_regr_loss_tests():

    ###--- Meta ---###
    epochs = 2
    batch_size = 100
    latent_dim = 3

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
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]


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

    # encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)
    # decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e, activation = activation)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d, activation = activation)

    ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    #ae_model = AE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    dataset_size_ae = len(ae_train_ds)
    dataset_size_ete = len(regr_train_ds)

    
    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    topo_loss = Topological(p = 2)

    regr_loss_term = RegrAdapter(Huber(delta = 1))

    ete_loss_terms = {
        'Reconstr': Weigh(reconstr_loss_term, weight = 1 - ete_regr_weight), 
        'Regr': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    ete_loss = Loss(CompositeLossTerm(loss_terms = ete_loss_terms))
    
    reconstr_loss = Loss(loss_term = reconstr_loss_term)
    

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

            loss_reconstr = reconstr_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
            )
            loss_topo = topo_loss(X_batch = X_batch, Z_batch = Z_batch)
            print(
                f'Loss Reconstr: {loss_reconstr.item()}\n'
                f'Loss Topological: {loss_topo.item()}'
            )
            loss_ae = loss_reconstr + loss_topo

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
    #plot_loss_tensor(observed_losses = ae_loss_obs.losses)
    #loss_observer.plot_agg_results()


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
        LossTermVisitorS(reconstr_loss_term, loss_name='L2_norm', eval_cfg = eval_cfg_reconstr),

        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitorS(regr_loss_term, loss_name = 'Huber', eval_cfg = eval_cfg_comp),
        LatentPlotVisitor(eval_cfg = eval_cfg_comp)
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
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )




"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ###--- AE in isolation ---###
    #AE_iso_training_procedure()
    

    ###--- VAE in isolation ---###
    #train_VAE_iso()
    #VAE_iso_training_procedure()
    #VAE_latent_visualisation()


    ###--- Compositions ---###
    #train_joint_seq_AE()
    #train_joint_seq_VAE()
    #train_seq_AE()
    #train_joint_epoch_wise_VAE()
    #train_joint_epoch_wise_VAE_recon()
    AE_joint_epoch_procedure()
    #VAE_joint_epoch_procedure()


    ###--- Baseline ---###
    #train_linear_regr()
    #train_deep_regr()


    ###--- Testing ---###
    #AE_regr_loss_tests()

    pass