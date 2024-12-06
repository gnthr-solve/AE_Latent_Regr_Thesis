
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path
from tqdm import tqdm

from data_utils import DatasetBuilder, SplitSubsetFactory, retrieve_metadata, time_col

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, ProductRegr, DNNRegr
from models import AE, GaussVAE, EnRegrComposite
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

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

from helper_tools import simple_timer
from helper_tools.plotting import plot_loss_tensor, plot_latent_with_reconstruction_error, plot_latent_with_attribute

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def AE_iso_training_procedure():
    
    ###--- Meta ---###
    epochs = 3
    batch_size = 50
    latent_dim = 3


    ###--- Dataset ---###
    #normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    normaliser = None

    dataset_builder = DatasetBuilder(
        kind = 'key',
        normaliser = normaliser,
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
    )
    
    dataset = dataset_builder.build_dataset()
    
    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models AE ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    # encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 5)

    # decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 5)

    # model = AE(encoder = encoder, decoder = decoder)

    #--- Models NaiveVAE ---#
    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5, activation='PReLU')
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5, activation='PReLU')

    #model = NaiveVAE_LogVar(encoder = encoder, decoder = decoder, input_dim = input_dim)
    #model = NaiveVAE_LogSigma(encoder = encoder, decoder = decoder)
    model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    #latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs, 
        dataset_size = dataset_size,
        batch_size = batch_size,
        members = ['Reconstruction'],
        name = 'AE Loss',
        aggregated = True,
    )

    
    ###--- Loss ---###
    reconstr_term = AEAdapter(LpNorm(p = 2))
    test_reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    loss_terms = {'Reconstruction': reconstr_term}
    reconstr_loss = Loss(CompositeLossTermObs(observer=loss_observer, **loss_terms))

    test_reconstr_loss = test_reconstr_term


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.5)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = reconstr_loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )

    # training_procedure.register_observers(
    #     #latent_observer, 
    #     #model_observer,
    # )
    
    training_procedure()


    ###--- Test Observers ---###
    loss_observer.plot_agg_results()
    #latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    X_test = dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    with torch.no_grad():

        Z_batch_hat, X_test_hat = model(X_test)

        loss_reconst = test_reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)

    print(
        f"After {epochs} epochs with {n_iterations} iterations each\n"
        f"Avg. Loss on testing subset: {loss_reconst.mean()}\n"
    )

    if latent_dim == 3:
        title = f'NVAE Normalised MinMax (epochs = {epochs})'
        plot_latent_with_reconstruction_error(
            latent_tensor = Z_batch_hat,
            loss_tensor = loss_reconst,
            title = title
        )




def VAE_iso_training_procedure():
    
    ###--- Meta ---###
    epochs = 3
    batch_size = 50
    latent_dim = 10


    ###--- Dataset ---###
    #normaliser = MinMaxNormaliser()
    normaliser = MinMaxEpsNormaliser(epsilon=1e-3)
    #normaliser = ZScoreNormaliser()
    #normaliser = None

    dataset_builder = DatasetBuilder(
        kind = 'key',
        normaliser = normaliser,
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
    )
    
    dataset = dataset_builder.build_dataset()
    

    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size = dataset_size,
        batch_size = batch_size,
        members = ['Log-Likelihood', 'KL-Divergence'],
        name = 'VAE Loss',
        aggregated = True,
    )


    ###--- Loss ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)
    #ll_term = Weigh(IndBetaLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    loss = Loss(CompositeLossTermObs(observer = loss_observer, **loss_terms))

    test_reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.5)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure.register_observers(
        latent_observer, 
        #model_observer,
    )
    
    training_procedure()


    ###--- Test Observers ---###
    loss_observer.plot_agg_results()
    #latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))

    
    ##--- Test Loss ---###
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    X_test = dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    Z_batch, infrm_dist_params, genm_dist_params = model(X_test)

    mu_l, logvar_l = infrm_dist_params.unbind(dim = -1)
    mu_r, logvar_r = genm_dist_params.unbind(dim = -1)

    var_l = torch.exp(logvar_l)
    var_r = torch.exp(logvar_r)

    #X_test_hat = model.reparameterise(genm_dist_params)
    X_test_hat = mu_r

    loss_reconst_test = test_reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    print(
        f'After {epochs} epochs with {len(dataloader)} iterations each\n'
        f'Avg. Loss on mean reconstruction in testing subset: \n{loss_reconst_test}\n'
        f'----------------------------------------\n\n'
        f'Inference M. mean:\n'
        f'max:\n{mu_l.max()}\n'
        f'min:\n{mu_l.min()}\n\n'
        f'Inference M. Var:\n'
        f'max:\n{var_l.max()}\n'
        f'min:\n{var_l.min()}\n'
        f'----------------------------------------\n\n'
        f'Generative M. mean:\n'
        f'max:\n{mu_r.max()}\n'
        f'min:\n{mu_r.min()}\n\n'
        f'Generative M. Var:\n'
        f'max:\n{var_r.max()}\n'
        f'min:\n{var_r.min()}\n'
        f'----------------------------------------\n\n'
    )




def VAE_latent_visualisation():
    
    from helper_tools import plot_latent_with_reconstruction_error

    ###--- Meta ---###
    epochs = 3
    batch_size = 20
    latent_dim = 3


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None

    dataset_builder = DatasetBuilder(
        kind = 'key',
        normaliser = normaliser,
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
    )
    
    dataset = dataset_builder.build_dataset()
    

    ###--- DataLoader ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_dataset = subset_factory.retrieve(kind = 'train', combine = True)

    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Observers ---###
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size = dataset_size,
        batch_size = batch_size,
        members = ['Log-Likelihood', 'KL-Divergence'],
        name = 'VAE Loss',
        aggregated = True,
    )


    ###--- Loss Construction ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    #--- Beta VAE Loss ---#
    beta = 10
    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()
    kld_term = Weigh(kld_term, weight = beta)

    #--- Composition ---#
    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    loss = Loss(CompositeLossTermObs(observer = loss_observer, **loss_terms))

    test_reconstr_loss = AEAdapter(RelativeLpNorm(p = 2))


    ###--- Optimizer & Scheduler ---###
    gamma = 0.5
    lr = 1e-3
    print(f"Learning rates: {[lr * gamma**epoch for epoch in range(epochs)]}")
    optimizer = Adam(model.parameters(), lr = lr)
    scheduler = ExponentialLR(optimizer, gamma = gamma)


    ###--- Training Procedure ---###
    training_procedure = AEIsoTrainingProcedure(
        train_dataloader = dataloader,
        ae_model = model,
        loss = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure.register_observers(
        latent_observer, 
        #model_observer,
    )
    
    training_procedure()


    ###--- Test Observers ---###
    loss_observer.plot_agg_results()
    #latent_observer.plot_dist_params_batch(lambda t: torch.max(torch.abs(t)))
    #model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    test_dataset = subset_factory.retrieve(kind = 'test', combine = True)
    indices = test_dataset.indices
    X_test = dataset.X_data[indices]
    mapping_idxs = X_test[:, 0].tolist()
    X_test = X_test[:, 1:]

    test_ds_metadata = retrieve_metadata(mapping_idxs, dataset.metadata_df)
    test_ds_metadata.loc[:, time_col] = pd.to_datetime(test_ds_metadata[time_col]).astype('int64') // 10**9
    

    with torch.no_grad():

        Z_batch, infrm_dist_params, genm_dist_params = model(X_test)

        mu_l, logvar_l = infrm_dist_params.unbind(dim = -1)
        mu_r, logvar_r = genm_dist_params.unbind(dim = -1)

        X_test_hat = mu_r

        loss_reconst_test = test_reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)

    print(
        f'After {epochs} epochs with {len(dataloader)} iterations each\n'
        f'Avg. Loss on mean reconstruction in testing subset: \n{loss_reconst_test.mean()}\n'
    )

    title = f'VAE Latent Representation (beta = {beta}, epochs = {epochs})'
    plot_latent_with_reconstruction_error(
        latent_tensor = mu_l,
        loss_tensor = loss_reconst_test,
        title = title
    )
    plot_latent_with_attribute(
        latent_tensor = mu_l,
        color_attr = test_ds_metadata[time_col],
        title = title
    )




def train_joint_seq_AE():

    ###--- Meta ---###
    epochs = 4
    batch_size = 50
    latent_dim = 10


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
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
    epochs = 1
    batch_size = 50
    latent_dim = 10


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
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

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5)

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    n_iterations_ae = len(dataloader_ae)
    n_iterations_regr = len(dataloader_regr)

    vae_model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations_ae, model = vae_model)
    #vae_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_ae)

    #regr_loss_obs = LossObserver(n_epochs = epochs, n_iterations = n_iterations_regr)


    ###--- Loss Terms ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTermObs(**vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))


    ###--- Losses ---###
    vae_loss = Loss(vae_loss_term)

    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)



    ###--- Optimizer & Scheduler ---###
    optimiser_vae = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
    ])

    optimiser_regr = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 5e-3},
    ])

    scheduler_vae = ExponentialLR(optimiser_vae, gamma = 0.5)
    scheduler_regr = ExponentialLR(optimiser_regr, gamma = 0.5)


    ###--- Training Loop AE---###
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        
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
            vae_model_obs(epoch = epoch, iter_idx = iter_idx, model = vae_model)
            #vae_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_vae)

        scheduler_vae.step()


    ###--- Training Loop Regr ---###
    for epoch in pbar:
        
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_regr.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)

            y_hat_batch = regressor(Z_batch)


            loss_regr = regr_loss(y_batch = y_batch, y_hat_batch = y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            optimiser_regr.step()

            #--- Observer Call ---#
            #regr_loss_obs(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_regr)


        scheduler_regr.step()


    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = vae_loss_obs.losses)
    #plot_loss_tensor(observed_losses = regr_loss_obs.losses)

    vae_model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))
    

    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    ae_test_ds = test_subsets['unlabelled']
    regr_test_ds = test_subsets['labelled']

    #--- Select data ---#
    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    #--- Apply VAE to labelled and unlabelled data ---#
    Z_batch_l, infrm_dist_params_l, genm_dist_params_l = vae_model(X_test_l)
    Z_batch_ul, infrm_dist_params_ul, genm_dist_params_ul = vae_model(X_test_ul)

    #--- Reconstruction (actually irrelevant here) ---#
    mu_r, _ = genm_dist_params_ul.unbind(dim = -1)
    X_test_ul_hat = mu_r

    loss_reconst = reconstr_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)

    #--- Reconstruction ---#
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




def train_joint_epoch_wise_VAE_recon():

    ###--- Meta ---###
    epochs = 2
    batch_size = 50
    latent_dim = 10


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
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
    
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size= dataset_size_ete,
        batch_size= batch_size,
        members = ['Reconstruction Term', 'Regression Term'],
        name = 'ETE Loss',
        aggregated = True,
    )


    ###--- Loss Terms ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    #kld_term = GaussianAnaKLDiv()
    kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_elbo_term = CompositeLossTermObs(**vae_loss_terms)

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
    ete_loss = Loss(CompositeLossTermObs(observer= loss_observer, **ete_loss_terms))
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
    loss_observer.plot_agg_results()


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
    epochs = 5
    batch_size = 25
    latent_dim = 3


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = MinMaxEpsNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
    dataset_builder = DatasetBuilder(
        kind = 'key',
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

    # encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)
    # decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4, activation='PReLU')
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4, activation='PReLU')

    #model = NaiveVAE(encoder = encoder, decoder = decoder)
    ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    #ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = LinearRegr(latent_dim = latent_dim)
    #regressor = ProductRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    dataset_size_ae = len(ae_train_ds)
    dataset_size_ete = len(regr_train_ds)

    ae_loss_obs = LossTermObserver(
        n_epochs = epochs, 
        dataset_size= dataset_size_ae,
        batch_size= batch_size,
        name = 'AE Loss',
        aggregated = True,
    )
    
    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size= dataset_size_ete,
        batch_size= batch_size,
        members = ['Reconstruction Term', 'Regression Term'],
        name = 'ETE Loss',
        aggregated = True,
    )


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight=0.05), 
        'Regression Term': Weigh(regr_loss_term, weight = 0.95),
    }

    ete_loss = Loss(CompositeLossTermObs(observer = loss_observer, **ete_loss_terms))
    #reconstr_loss = Loss(Observe(observer = ae_loss_obs, loss_term = reconstr_loss_term))
    reconstr_loss = Loss(loss_term = reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.5)


    ###--- Training Procedure ---###
    training_procedure = JointEpochTrainingProcedure(
        ae_train_dataloader = dataloader_ae,
        regr_train_dataloader = dataloader_regr,
        ae_model = ae_model,
        regr_model = regressor,
        ae_loss = reconstr_loss,
        ete_loss = ete_loss, 
        optimizer = optimiser,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure()

    ###--- Plot Observations ---###
    #plot_loss_tensor(observed_losses = ae_loss_obs.losses)
    loss_observer.plot_agg_results()


    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    ae_test_ds = test_subsets['unlabelled']
    regr_test_ds = test_subsets['labelled']

    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():

        Z_batch_ul, X_test_ul_hat = ae_model(X_test_ul)

        loss_reconst = reconstr_loss_term(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)

        Z_batch_l, X_test_l_hat = ae_model(X_test_l)
        y_test_l_hat = regressor(Z_batch_l)

        loss_regr = regr_loss_term(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconst.mean()}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr.mean()}\n"
    )

    if latent_dim == 3:
        title = f'NVAE Normalised (epochs = {epochs})'
        plot_latent_with_reconstruction_error(
            latent_tensor = Z_batch_ul,
            loss_tensor = loss_reconst,
            title = title
        )
        title = f'NVAE Normalised Regr(epochs = {epochs})'
        plot_latent_with_reconstruction_error(
            latent_tensor = Z_batch_l,
            loss_tensor = loss_regr,
            title = title
        )




def VAE_joint_epoch_procedure():

    ###--- Meta ---###
    epochs = 5
    batch_size = 25
    latent_dim = 10


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None
    
    dataset_builder = DatasetBuilder(
        kind = 'key',
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

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3, activation = 'Softplus')

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 3, activation = 'Softplus')

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Observation Test Setup ---###
    dataset_size_ete = len(regr_train_ds)

    loss_observer = CompositeLossTermObserver(
        n_epochs = epochs,
        dataset_size= dataset_size_ete,
        batch_size= batch_size,
        members = ['Reconstruction Term', 'Regression Term'],
        name = 'ETE Loss',
        aggregated = True,
    )

    ###--- Loss Terms ---###
    ll_term = Weigh(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTermObs(**vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(vae_loss_term, weight=0.05), 
        'Regression Term': Weigh(regr_loss_term, weight = 0.95),
    }


    ###--- Losses ---###
    #--- For Training ---#
    vae_loss = Loss(vae_loss_term)
    ete_loss = Loss(CompositeLossTermObs(observer = loss_observer, **ete_loss_terms))

    #--- For Testing ---#
    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)



    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.5)


    ###--- Training Procedure ---###
    training_procedure = JointEpochTrainingProcedure(
        ae_train_dataloader = dataloader_ae,
        regr_train_dataloader = dataloader_regr,
        ae_model = vae_model,
        regr_model = regressor,
        ae_loss = vae_loss,
        ete_loss = ete_loss, 
        optimizer = optimiser,
        scheduler = scheduler,
        epochs = epochs,
    )

    training_procedure()


    ###--- Plot Observations ---###
    loss_observer.plot_agg_results()


    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    ae_test_ds = test_subsets['unlabelled']
    regr_test_ds = test_subsets['labelled']

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




def train_linear_regr():

    ###--- Meta ---###
    epochs = 10
    batch_size = 30


    ###--- Dataset ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = None

    dataset_builder = DatasetBuilder(
        kind = 'key',
        normaliser = normaliser,
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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
    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.9)


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
    epochs = 10
    batch_size = 30


    ###--- Dataset ---###
    #normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    normaliser = None

    dataset_builder = DatasetBuilder(
        kind = 'key',
        normaliser = normaliser,
        exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
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
    regressor = DNNRegr(input_dim = input_dim, n_layers = 4, activation = 'Softplus')

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
    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.9)


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
    #train_joint_epoch_wise_VAE()
    #train_joint_epoch_wise_VAE_recon()
    #AE_joint_epoch_procedure()
    #VAE_joint_epoch_procedure()


    ###--- Baseline ---###
    #train_linear_regr()
    train_deep_regr()

    pass