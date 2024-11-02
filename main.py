
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch import Tensor

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from tqdm import tqdm

from datasets import TensorDataset, DataFrameDataset

from models.encoders import (
    SimpleLinearReluEncoder,
    LinearReluEncoder,
    GeneralLinearReluEncoder,
)

from models.decoders import (
    SimpleLinearReluDecoder,
    LinearReluDecoder,
    GeneralLinearReluDecoder,
)

from models.var_encoders import VarEncoder
from models.var_decoders import VarDecoder

from models.regressors import LinearRegr
from models import SimpleAutoencoder, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE, NaiveVAESigma, NaiveVAELogSigma

from loss import (
    Loss,
    CompositeLossTerm,
    CompositeLossTermAlt,
    WeightedLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    HuberOwn,
)

from loss.adapters_decorators import AEAdapter, RegrAdapter
from loss.loss_classes import NegativeELBOLoss
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL

from observers.ae_param_observer import AEParameterObserver
from observers import LossObserver, ModelObserver, VAELatentObserver

from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from helper_tools import plot_loss_tensor, get_valid_batch_size, plot_training_characteristics


"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def train_AE_iso_observer_test():

    #from observers.model_observer import ModelObserver

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')

    #X_data = torch.tensor(data=X_data.data, dtype=torch.float64)
    
    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    batch_size = 50
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models AE ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    # encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    # decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    # model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    #--- Models NaiveVAE ---#
    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)

    #model = NaiveVAE(encoder = encoder, decoder = decoder)
    model = NaiveVAELogSigma(encoder = encoder, decoder = decoder)
    #model = NaiveVAESigma(encoder = encoder, decoder = decoder)

    #initialize_weights(model)

    
    ###--- Loss ---###
    #reconstr_term = AEAdapter(LpNorm(p = 2))
    reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    loss_terms = {'Reconstruction': reconstr_term}
    reconstr_loss = Loss(CompositeLossTerm(**loss_terms))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.5)


    ###--- Meta ---###
    epochs = 2
    pbar = tqdm(range(epochs))


    ###--- Observation Test Setup ---###
    n_iterations = len(dataloader)
    loss_obs_tensor = torch.zeros(size = (epochs, n_iterations))
    model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations, model = model)

    ###--- Training Loop ---###
    for epoch in pbar:
        
        for b_idx, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            loss_obs_tensor[epoch, b_idx] = loss_reconst.detach()

            #--- Backward Pass ---#
            loss_reconst.backward()

            optimizer.step()

            #--- Observer Call ---#
            model_obs(epoch = epoch, iter_idx = b_idx, model = model)


        scheduler.step()


    plot_loss_tensor(observed_losses = loss_obs_tensor)
    #model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))

    ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]
    X_test_hat = model(X_test)

    loss_reconst = reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    print(
        f"After {epochs} epochs with {len(dataloader)} iterations each\n"
        f"Avg. Loss on testing subset: {loss_reconst}\n"
    )




def train_AE_NVAE_iso():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    batch_size = 50
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models AE ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    #encoder = LinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim)
    # encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    # #decoder = LinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim)
    # decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    # model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    #--- Models NaiveVAE ---#
    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    #model = NaiveVAE(encoder = encoder, decoder = decoder)
    model = NaiveVAESigma(encoder = encoder, decoder = decoder)

    ###--- Loss ---###
    #reconstr_term = AEAdapter(LpNorm(p = 2))
    reconstr_term = AEAdapter(RelativeLpNorm(p = 2))

    loss_terms = {'Reconstruction': reconstr_term}
    reconstr_loss = Loss(CompositeLossTerm(**loss_terms))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)


    ###--- Meta ---###
    epochs = 2
    pbar = tqdm(range(epochs))

    observer = AEParameterObserver()


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            #print(f"{it}_{b_ind+1}/{epochs}")
            observer(loss = loss_reconst, ae_model = model)


            optimizer.step()


        scheduler.step()

    observer.plot_results()


    ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]
    X_test_hat = model(X_test)

    loss_reconst = reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    print(
        f"After {epochs} epochs with {len(dataloader)} iterations each\n"
        f"Avg. Loss on testing subset: {loss_reconst}\n"
    )




def train_VAE_iso():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ##--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    batch_size = 100
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Loss ---###
    ll_term = WeightedLossTerm(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    loss = Loss(CompositeLossTerm(**loss_terms))
    

    test_reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)


    ###--- Meta ---###
    epochs = 2
    pbar = tqdm(range(epochs))

    #observer = AEParameterObserver()
    observer = VAELatentObserver(n_epochs=epochs, dataset_size=len(train_dataset), batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = model(X_batch)

            loss_reconst = loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )

    
            #--- Backward Pass ---#
            loss_reconst.backward()

            #print(f"{it}_{b_ind+1}/{epochs}")
            #observer(loss = loss_reconst, ae_model = model)
            observer(epoch = it, iter_idx = b_ind, infrm_dist_params = infrm_dist_params)

            optimizer.step()


        scheduler.step()

    #observer.plot_results()
    observer.plot_dist_params_batch()

    # ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
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




def VAE_iso_training_procedure_test():

    from training.training_procedure import VAEIsoTrainingProcedure
    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ##--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    batch_size = 100
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 5
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Loss ---###
    ll_term = WeightedLossTerm(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    loss = Loss(CompositeLossTerm(**loss_terms))

    test_reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.5)


    ###--- Meta ---###
    epochs = 4
    n_iterations = len(dataloader)
    dataset_size = len(train_dataset)

    ###--- Set up Observers ---###
    loss_observer = LossObserver(n_epochs = epochs, n_iterations = n_iterations)
    latent_observer = VAELatentObserver(n_epochs=epochs, dataset_size=dataset_size, batch_size=batch_size, latent_dim=latent_dim, n_dist_params=2)
    model_observer = ModelObserver(n_epochs = epochs, n_iterations = n_iterations, model = model)

    ###--- Training Procedure ---###
    training_procedure = VAEIsoTrainingProcedure(
        train_dataloader = dataloader,
        vae_model = model,
        loss = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        epochs = epochs,
        #batch_size = batch_size
    )

    training_procedure.register_observers(loss_observer, latent_observer, model_observer)
    
    training_procedure()


    ###--- Test Observers ---###
    plot_loss_tensor(loss_observer.losses)
    latent_observer.plot_dist_params_batch(functional = torch.max)
    model_observer.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
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




def train_joint_seq_AE():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### #NOTE: Need to split dataset in parts with label and parts w.o. label before split?
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 50
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    regressor = LinearRegr(latent_dim = latent_dim)

    regr_model = EnRegrComposite(encoder = encoder, regressor = regressor)
    ae_model = SimpleAutoencoder(encoder = encoder, decoder = decoder)


    ###--- Losses ---###
    #reconstr_loss = Loss(LpNorm(p = 2))
    reconstr_loss = Loss(AEAdapter(RelativeLpNorm(p = 2)))
    regr_loss = Loss(RegrAdapter(Huber(delta = 1)))


    ###--- Optimizer & Scheduler ---###
    optimizer_ae = Adam(ae_model.parameters(), lr = 1e-2)
    scheduler_ae = ExponentialLR(optimizer_ae, gamma = 0.1)

    optimizer_regr = Adam(regr_model.parameters(), lr = 1e-2)
    scheduler_regr = ExponentialLR(optimizer_regr, gamma = 0.1)


    ###--- Meta ---###
    epochs = 1
    pbar = tqdm(range(epochs))

    observer = AEParameterObserver()


    ###--- Training Loop AE---###
    for it in pbar:
        
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_ae.zero_grad()
            
            X_hat_batch = ae_model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            #print(f"{it}_{b_ind+1}/{epochs}")
            observer(loss = loss_reconst, ae_model = ae_model)


            optimizer_ae.step()


        scheduler_ae.step()

    observer.plot_results()


    ###--- Training Loop Regr ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer_regr.zero_grad()
            
            y_hat_batch = regr_model(X_batch)

            loss_regr = regr_loss(y_batch = y_batch, y_hat_batch = y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            print(
                f"{it}_{b_ind+1}/{epochs} Regr. Loss:\n"
                f"{loss_regr.tolist()}"
            )

            optimizer_regr.step()


        scheduler_regr.step()


    ###--- Test Loss ---###
    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    X_test_ae = dataset.X_data[ae_test_ds.indices]
    X_test_regr = dataset.X_data[regr_test_ds.indices]
    y_test_regr =dataset.y_data[regr_test_ds.indices]
    X_test_ae = X_test_ae[:, 1:]
    X_test_regr = X_test_regr[:, 1:]
    y_test_regr = y_test_regr[:, 1:]

    X_test_hat = ae_model(X_test_ae)
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

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### #NOTE: Need to split dataset in parts with label and parts w.o. label before split?
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 50
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 5
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 5)

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Loss Terms ---###
    ll_term = WeightedLossTerm(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTermAlt(print_losses = True, **vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
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


    ###--- Meta ---###
    epochs = 3
    pbar = tqdm(range(epochs))

    observer = AEParameterObserver()


    ###--- Training Loop AE---###
    for it in pbar:
        
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
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

            #print(f"{it}_{b_ind+1}/{epochs}")
            observer(loss = loss_vae, ae_model = vae_model)


            optimiser_vae.step()


        scheduler_vae.step()

    observer.plot_results()


    ###--- Training Loop Regr ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser_regr.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)

            y_hat_batch = regressor(Z_batch)


            loss_regr = regr_loss(y_batch = y_batch, y_hat_batch = y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            print(
                f"{it}_{b_ind+1}/{epochs} Regr. Loss:\n"
                f"{loss_regr.tolist()}"
            )

            optimiser_regr.step()


        scheduler_regr.step()


    ###--- Test Loss ---###
    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

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




def train_joint_epoch_wise_AE():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### #NOTE: Need to split dataset in parts with label and parts w.o. label before split?
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 50
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    print(len(dataloader_ae), len(dataloader_regr))

    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)
    decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    # encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)
    # decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    # #model = NaiveVAE(encoder = encoder, decoder = decoder)
    # model = NaiveVAESigma(encoder = encoder, decoder = decoder)


    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': WeightedLossTerm(reconstr_loss_term, weight=0.1), 
        'Regression Term': WeightedLossTerm(regr_loss_term, weight = 0.9),
    }

    ete_loss = Loss(CompositeLossTermAlt(print_losses = True, **ete_loss_terms))
    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.5)


    ###--- Meta ---###
    epochs = 3
    pbar = tqdm(range(epochs))

    from observers.loss_observer import LossObserverAlpha
    loss_observer = LossObserverAlpha('Reconstruction', 'End-To-End')


    ###--- Training Loop Joint---###
    for it in pbar:
        
        ###--- Training Loop AE---###
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            X_hat_batch = decoder(encoder(X_batch))
            #X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            loss_observer('Reconstruction', loss_reconst)
            # print(
            #     f"Epoch {it + 1}/{epochs}; AE iteration {b_ind + 1}/{len(dataloader_ae)}\n"
            #     f"Loss Reconstruction: {loss_reconst.item()}"
            # )

            optimiser.step()


        ###--- Training Loop End-To-End ---###
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch = encoder(X_batch)
            X_hat_batch = decoder(Z_batch)
            y_hat_batch = regressor(Z_batch)

            #print(Z_batch.shape, X_hat_batch.shape, y_hat_batch.shape)

            loss_ete_weighted = ete_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            loss_observer('End-To-End', loss_ete_weighted)
            # print(
            #     f"Epoch {it + 1}/{epochs}; ETE iteration {b_ind + 1}/{len(dataloader_regr)}\n"
            #     f"Loss End-To-End: {loss_ete_weighted.item()}"
            # )

            optimiser.step()


        scheduler.step()

    loss_observer.plot_results()


    ###--- Test Loss ---###
    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    X_test_ul_hat = decoder(encoder((X_test_ul)))

    loss_reconst = reconstr_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)

    y_test_l_hat = regressor(encoder(X_test_l))

    loss_regr = regr_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)
    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_ae)} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconst}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )




def train_joint_epoch_wise_VAE():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### #NOTE: Need to split dataset in parts with label and parts w.o. label before split?
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 50
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    print(len(dataloader_ae), len(dataloader_regr))

    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 8)

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Loss Terms ---###
    ll_term = WeightedLossTerm(GaussianDiagLL(), weight = -1)

    kld_term = GaussianAnaKLDiv()
    #kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_loss_term = CompositeLossTermAlt(print_losses = True, **vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': WeightedLossTerm(vae_loss_term, weight=0.2), 
        'Regression Term': WeightedLossTerm(regr_loss_term, weight = 0.8),
    }

    ###--- Losses ---###
    vae_loss = Loss(vae_loss_term)
    ete_loss = Loss(CompositeLossTermAlt(print_losses = True, **ete_loss_terms))
    reconstr_loss = Loss(reconstr_loss_term)
    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.5)


    ###--- Meta ---###
    epochs = 3
    pbar = tqdm(range(epochs))

    from observers.loss_observer import LossObserverAlpha
    loss_observer = LossObserverAlpha('Reconstruction', 'End-To-End')


    ###--- Training Loop Joint---###
    for it in pbar:
        
        ###--- Training Loop AE---###
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)

            loss_reconst = vae_loss(
                X_batch = X_batch,
                Z_batch = Z_batch,
                genm_dist_params = genm_dist_params,
                infrm_dist_params = infrm_dist_params,
            )


            #--- Backward Pass ---#
            loss_reconst.backward()

            loss_observer('Reconstruction', loss_reconst)
            # print(
            #     f"Epoch {it + 1}/{epochs}; AE iteration {b_ind + 1}/{len(dataloader_ae)}\n"
            #     f"Loss Reconstruction: {loss_reconst.item()}"
            # )

            optimiser.step()


        ###--- Training Loop End-To-End ---###
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, infrm_dist_params, genm_dist_params = vae_model(X_batch)
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

            loss_observer('End-To-End', loss_ete_weighted)
            # print(
            #     f"Epoch {it + 1}/{epochs}; ETE iteration {b_ind + 1}/{len(dataloader_regr)}\n"
            #     f"Loss End-To-End: {loss_ete_weighted.item()}"
            # )

            optimiser.step()


        scheduler.step()

    loss_observer.plot_results()


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




def train_joint_epoch_wise_VAE_recon():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### #NOTE: Need to split dataset in parts with label and parts w.o. label before split?
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    ae_train_ds = subsets['train_unlabeled']
    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 100
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    #print(len(dataloader_ae), len(dataloader_regr))

    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    vae_model = GaussVAE(encoder = encoder, decoder = decoder)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Loss Terms ---###
    ll_term = WeightedLossTerm(GaussianDiagLL(), weight = -1)

    #kld_term = GaussianAnaKLDiv()
    kld_term = GaussianMCKLDiv()

    vae_loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}
    vae_elbo_term = CompositeLossTermAlt(print_losses = True, **vae_loss_terms)

    #reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': WeightedLossTerm(reconstr_loss_term, weight=0.1), 
        'Regression Term': WeightedLossTerm(regr_loss_term, weight = 0.9),
    }

    ###--- Losses ---###
    vae_loss = Loss(vae_elbo_term)
    ete_loss = Loss(CompositeLossTermAlt(print_losses = True, **ete_loss_terms))
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


    ###--- Meta ---###
    epochs = 6
    pbar = tqdm(range(epochs))

    from observers.loss_observer import LossObserverAlpha
    loss_observer = LossObserverAlpha('VAE-Loss', 'End-To-End')


    ###--- Training Loop Joint---###
    for it in pbar:
        
        ###--- Training Loop AE---###
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
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

            loss_observer('VAE-Loss', loss_vae)
            # print(
            #     f"Epoch {it + 1}/{epochs}; AE iteration {b_ind + 1}/{len(dataloader_ae)}\n"
            #     f"Loss Reconstruction: {loss_reconst.item()}"
            # )

            optimiser_vae.step()


        ###--- Training Loop End-To-End ---###
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
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

            loss_observer('End-To-End', loss_ete_weighted)
            # print(
            #     f"Epoch {it + 1}/{epochs}; ETE iteration {b_ind + 1}/{len(dataloader_regr)}\n"
            #     f"Loss End-To-End: {loss_ete_weighted.item()}"
            # )

            optimiser_ete.step()


        scheduler_vae.step()
        scheduler_ete.step()


    loss_observer.plot_results()


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




def train_baseline():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    normaliser = MinMaxNormaliser()
    #normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)


    ###--- Dataset---### 
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    subsets = subset_factory.create_splits()

    regr_train_ds = subsets['train_labeled']


    ###--- DataLoader ---###
    batch_size = 50
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    regressor = LinearRegr(latent_dim = input_dim)


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(HuberOwn(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(regr_loss_term)


    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.5)


    ###--- Meta ---###
    epochs = 3
    pbar = tqdm(range(epochs))

    from observers.loss_observer import LossObserverAlpha
    loss_observer = LossObserverAlpha('Regression')


    ###--- Training Loop Joint---###
    for it in pbar:
        
        ###--- Training Loop End-To-End ---###
        for b_ind, (X_batch, y_batch) in enumerate(dataloader_regr):
            
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

            loss_observer('Regression', loss_regr)
            print(
                f"Epoch {it + 1}/{epochs}; iteration {b_ind + 1}/{len(dataloader_regr)}\n"
                f"Loss Regression: {loss_regr.item()}"
            )

            optimiser.step()


        scheduler.step()


    loss_observer.plot_results()


    ###--- Test Loss ---###
    regr_test_ds = subsets['test_labeled']

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    y_test_l_hat = regressor(X_test_l)

    loss_regr = regr_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)
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

    ###--- AE in isolation ---###
    #main_test_view_DataFrameDS()
    #main_test_view_TensorDS()
    #train_AE_iso_observer_test()
    #train_AE_NVAE_iso()

    ###--- VAE in isolation ---###
    #train_VAE_iso()
    #VAE_iso_training_procedure_test()

    ###--- Compositions ---###
    #train_joint_seq_AE()
    #train_joint_seq_VAE()
    train_joint_epoch_wise_AE()
    #train_joint_epoch_wise_VAE()
    #train_joint_epoch_wise_VAE_recon()

    ###--- Baseline ---###
    #train_baseline()

    pass