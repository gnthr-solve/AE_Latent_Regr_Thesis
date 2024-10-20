
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
from models import SimpleAutoencoder, GaussVAE, NaiveVAE, NaiveVAESigma, EnRegrComposite

from loss import (
    WeightedCompositeLoss,
    MeanLpLoss,
    RelativeMeanLpLoss,
    HuberLoss,
    RelativeHuberLoss,
)
from loss.composite_losses import VAELoss
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_reconstr import GaussianDiagRLT

from ae_param_observer import AEParameterObserver
from loss_observer import LossObserver
from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from helper_tools import get_valid_batch_size, plot_training_characteristics


"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def main_test_view_DataFrameDS():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    joint_data_df = pd.read_csv(data_dir / "data_joint.csv")

    print(
        f"Shape joint_data_df: \n {joint_data_df.shape}\n"
        f"joint_data_df dtypes: \n {joint_data_df.dtypes}\n"
    )


    ###--- Dataset & DataLoader ---###
    batch_size = 32

    dataset = DataFrameDataset(joint_data_df)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    print(
        f"\nDataSet properties:\n"
        f"----------------------------------------------\n"
        f"X data dimensions: \n{dataset.X_dim}\n"
        f"y data dimensions: \n{dataset.y_dim}\n"
        f"Size of (X) dataset: \n{len(dataset)}\n"
        f"Size of available y data: \n{get_valid_batch_size(dataset.y_data)}\n"
        f"----------------------------------------------\n"
    )


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim[0] - 1
    print(f"input_dim: {input_dim}")

    encoder = SimpleLinearReluEncoder(latent_dim = latent_dim)
    decoder = SimpleLinearReluDecoder(latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    print(
        f"Model: \n {model}\n"
        f"Model Parameters: \n {model.parameters}\n"
        f"Model children: \n {model.children}\n"
        f"Model modules: \n {model.modules}\n"
    )


    ###--- Loss ---###
    #
    #reconstr_loss = MeanLpLoss(p = 2)
    reconstr_loss = RelativeMeanLpLoss(p = 2)


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.9)
    
    ###--- Meta ---###
    epochs = 10
    pbar = tqdm(range(epochs))


    ###--- Meta ---###
    epochs = 5
    pbar = tqdm(range(epochs))


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader):
            
            X_batch: Tensor = X_batch[:, 1:]
            y_batch: Tensor = y_batch[:, 1:]

            print(
                f"Shape X_batch: \n {X_batch.shape}\n"
                f"X_batch: \n {X_batch}\n"
                f"Shape y_batch: \n {y_batch.shape}\n"
                f"y_batch: \n {y_batch}\n"
            )

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch, X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            print(f"{it}_{b_ind+1}/{epochs} Parameters:")
            for name, param in model.named_parameters():
                print(f'{name} value:\n {param.data}')
                print(f'{name} grad:\n {param.grad}')
                
                if torch.isnan(param.data).any():
                    print(f"{name} contains NaN values")
                    raise StopIteration
                
                if torch.isinf(param.data).any():
                    print(f"{name} contains Inf values")
                    raise StopIteration

            print(f"Reconstruction Loss: {loss_reconst}")
            if loss_reconst.isnan():
                print("NaN Loss!")
                break

            optimizer.step()
            
            break


        scheduler.step()

        break




def main_test_view_TensorDS():

    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv")

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')


    ###--- Normalise ---###
    #normaliser = MinMaxNormaliser()
    normaliser = ZScoreNormaliser()
    #normaliser = RobustScalingNormaliser()

    with torch.no_grad():
        X_data[:, 1:] = normaliser.normalise(X_data[:, 1:])
        print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        print(X_data.shape)

    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    print(
        f"\nDataSet properties:\n"
        f"----------------------------------------------\n"
        f"X data dimensions: \n{dataset.X_dim}\n"
        f"y data dimensions: \n{dataset.y_dim}\n"
        f"Size of (X) dataset: \n{len(dataset)}\n"
        f"Size of available y data: \n{get_valid_batch_size(dataset.y_data)}\n"
        f"----------------------------------------------\n"
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    batch_size = 50
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = LinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim)
    #encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    decoder = LinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim)
    #decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    print(
        f"Model: \n {model}\n"
        f"Model Parameters: \n {model.parameters}\n"
        f"Model children: \n {model.children}\n"
        f"Model modules: \n {model.modules}\n"
    )


    ###--- Loss ---###
    #
    #reconstr_loss = MeanLpLoss(p = 2)
    reconstr_loss = RelativeMeanLpLoss(p = 2)


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)

    
    ###--- Meta ---###
    epochs = 5
    pbar = tqdm(range(epochs))


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader):
            
            X_batch: Tensor = X_batch[:, 1:]
            y_batch: Tensor = y_batch[:, 1:]

            print(
                f"Shape X_batch: \n {X_batch.shape}\n"
                f"X_batch: \n {X_batch}\n"
                f"Shape y_batch: \n {y_batch.shape}\n"
                f"y_batch: \n {y_batch}\n"
            )

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch, X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            print(f"{it}_{b_ind+1}/{epochs} Parameters:")
            for name, param in model.named_parameters():
                print(f'{name} value:\n {param.data}')
                print(f'{name} grad:\n {param.grad}')
                
                if torch.isnan(param.data).any():
                    print(f"{name} contains NaN values")
                    raise StopIteration
                
                if torch.isinf(param.data).any():
                    print(f"{name} contains Inf values")
                    raise StopIteration

            print(f"Reconstruction Loss: {loss_reconst}")
            if loss_reconst.isnan():
                print("NaN Loss!")
                break

            optimizer.step()
            
            break


        scheduler.step()

        break




def train_AE_NVAE_iso():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv")

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
    #reconstr_loss = MeanLpLoss(p = 2)
    reconstr_loss = RelativeMeanLpLoss(p = 2)


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)


    ###--- Meta ---###
    epochs = 5
    pbar = tqdm(range(epochs))

    observer = AEParameterObserver()


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch, X_hat_batch)

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

    loss_reconst = reconstr_loss(X_test, X_test_hat)
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

    metadata_df = pd.read_csv(data_dir / "metadata.csv")

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


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = 4)

    model = GaussVAE(encoder = encoder, decoder = decoder)


    ###--- Loss ---###
    reconstr_loss = GaussianDiagRLT()

    #kld_loss = GaussianAnaKLDiv()
    kld_loss = GaussianMCKLDiv()

    loss = VAELoss(reconstr_loss = reconstr_loss, kl_div_loss = kld_loss)

    test_reconstr_loss = RelativeMeanLpLoss(p = 2)

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.1)


    ###--- Meta ---###
    epochs = 5
    pbar = tqdm(range(epochs))

    observer = AEParameterObserver()


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
            observer(loss = loss_reconst, ae_model = model)


            optimizer.step()


        scheduler.step()

    observer.plot_results()


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

    loss_reconst_test = test_reconstr_loss(x_batch = X_test, x_hat_batch = X_test_hat)
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




def train_joint_seq():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv")

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

    encoder = LinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim)
    #encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    decoder = LinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim)
    #decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    regressor = LinearRegr(latent_dim = latent_dim)

    regr_model = EnRegrComposite(encoder = encoder, regressor = regressor)
    ae_model = SimpleAutoencoder(encoder = encoder, decoder = decoder)


    ###--- Losses ---###
    #reconstr_loss = MeanLpLoss(p = 2)
    reconstr_loss = RelativeMeanLpLoss(p = 2)
    regr_loss = HuberLoss(delta = 1)


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

            loss_reconst = reconstr_loss(X_batch, X_hat_batch)

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

            loss_regr = regr_loss(y_batch, y_hat_batch)

            #--- Backward Pass ---#
            loss_regr.backward()

            print(
                f"{it}_{b_ind+1}/{epochs} Regr. Loss:\n"
                f"{loss_regr.tolist()}"
            )

            optimizer_regr.step()


        scheduler_regr.step()


    ###--- Test Loss ---###
    # X_test = test_dataset.dataset.X_data[test_dataset.indices]
    # y_test = test_dataset.dataset.y_data[test_dataset.indices]
    # X_test = X_test[:, 1:]
    # y_test = y_test[:, 1:]
    # X_test_hat = ae_model(X_test)

    # loss_reconst = reconstr_loss(X_test, X_test_hat)
    # print(
    #     f"After {epochs} epochs with {len(dataloader)} iterations each\n"
    #     f"Avg. Loss on testing subset: {loss_reconst}\n"
    # )




def train_joint_epoch_wise():

    from datasets import SplitSubsetFactory

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    metadata_df = pd.read_csv(data_dir / "metadata.csv")

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

    encoder = LinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim)
    #encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    decoder = LinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim)
    #decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    regressor = LinearRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    #reconstr_loss = MeanLpLoss(p = 2)
    reconstr_loss = RelativeMeanLpLoss(p = 2)
    #regr_loss = HuberLoss(delta = 1)
    regr_loss = RelativeHuberLoss(delta = 1)

    weighted_loss = WeightedCompositeLoss(
        loss_regr = regr_loss,
        loss_reconstr = reconstr_loss,
        w_regr = 0.8,
        w_reconstr = 0.2
    )

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': decoder.parameters(), 'lr': 1e-3},
        {'params': regressor.parameters(), 'lr': 1e-2},
    ])

    scheduler = ExponentialLR(optimiser, gamma = 0.9)


    ###--- Meta ---###
    epochs = 4
    pbar = tqdm(range(epochs))

    loss_observer = LossObserver('Reconstruction', 'End-To-End')


    ###--- Training Loop Joint---###
    for it in pbar:
        
        ###--- Training Loop AE---###
        for b_ind, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            X_hat_batch = decoder(encoder(X_batch))
            # if it > 0:
            #     print(X_hat_batch[:5])

            loss_reconst = reconstr_loss(X_batch, X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            loss_observer('Reconstruction', loss_reconst)
            print(
                f"Epoch {it + 1}/{epochs}; AE iteration {b_ind + 1}/{len(dataloader_ae)}\n"
                f"Loss Reconstruction: {loss_reconst.item()}"
            )

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

            loss_ete_weighted = weighted_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            loss_observer('End-To-End', loss_ete_weighted)
            print(
                f"Epoch {it + 1}/{epochs}; ETE iteration {b_ind + 1}/{len(dataloader_regr)}\n"
                f"Loss End-To-End: {loss_ete_weighted.item()}"
            )

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

    loss_reconst = reconstr_loss(X_test_ul, X_test_ul_hat)

    y_test_l_hat = regressor(encoder(X_test_l))

    loss_regr = regr_loss(y_test_l, y_test_l_hat)
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




"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    #--- Test Main Functions (AE only) ---#
    #main_test_view_DataFrameDS()
    #main_test_view_TensorDS()
    #train_AE_NVAE_iso()
    train_VAE_iso()
    #train_joint_seq()
    #train_joint_epoch_wise()

    pass