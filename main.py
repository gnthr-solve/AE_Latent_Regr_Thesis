
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from tqdm import tqdm

from datasets import TensorDataset, DataFrameDataset
from models import (
    SimpleEncoder, 
    SimpleLinearReluEncoder,
    SimpleDecoder,
    SimpleLinearReluDecoder, 
    SimpleAutoencoder, 
    SimpleLoss,
    MeanLpLoss,
    RelativeMeanLpLoss,
)

from ae_param_observer import AEParameterObserver
from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from helper_tools import get_valid_batch_size, plot_training_characteristics


"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def main_test_view():

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

    encoder = SimpleEncoder(input_dim = input_dim, latent_dim = latent_dim)
    decoder = SimpleDecoder(input_dim = input_dim, latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    print(
        f"Model: \n {model}\n"
        f"Model Parameters: \n {model.parameters}\n"
        f"Model children: \n {model.children}\n"
        f"Model modules: \n {model.modules}\n"
    )

    ###--- Meta ---###
    epochs = 10
    pbar = tqdm(range(epochs))


    ###--- Training Loop ---###
    for it in pbar:
        
        for X_batch, y_batch in dataloader:
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            print(
                f"Shape X_batch: \n {X_batch.shape}\n"
                f"X_batch: \n {X_batch}\n"
                f"Shape y_batch: \n {y_batch.shape}\n"
                f"y_batch: \n {y_batch}\n"
            )

            break            

        break




def main_test_simple():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    joint_data_df = pd.read_csv(data_dir / "data_joint.csv")


    ###--- Dataset & DataLoader ---###
    batch_size = 32

    dataset = DataFrameDataset(joint_data_df)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    

    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim[0] - 1

    encoder = SimpleEncoder(input_dim = input_dim, latent_dim = latent_dim)
    decoder = SimpleDecoder(input_dim = input_dim, latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    loss = SimpleLoss()


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.99)


    ###--- Meta ---###
    epochs = 1
    pbar = tqdm(range(epochs))


    ###--- Training Loop ---###
    for it in pbar:
        
        for b_ind, (X_batch, y_batch) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            if torch.isnan(X_batch).any():
                continue

            print(
                #f"Shape X_batch: \n {X_batch.shape}\n"
                #f"X_batch: \n {X_batch}\n"
                #f"Shape y_batch: \n {y_batch.shape}\n"
                #f"y_batch: \n {y_batch}\n"
            )

            #--- Forward Pass ---#
            optimizer.zero_grad()

            X_hat_batch = model(X_batch)

            loss_reconst = loss(X_batch, X_hat_batch)


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


        scheduler.step()




def main_test_lin_relu():

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    joint_data_df = pd.read_csv(data_dir / "data_joint.csv")


    ###--- Dataset & DataLoader ---###
    batch_size = 200

    dataset = DataFrameDataset(joint_data_df)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim[0] - 1
    print(f"input_dim: {input_dim}")

    encoder = SimpleLinearReluEncoder(latent_dim = latent_dim)
    decoder = SimpleLinearReluDecoder(latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    #reconstr_loss = SimpleLoss()
    reconstr_loss = MeanLpLoss(p = 2)
    #reconstr_loss = RelativeMeanLpLoss(p = 2)

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.9)


    ###--- Meta ---###
    epochs = 1
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

            print(f"{it}_{b_ind+1}/{epochs}")
            observer(loss = loss_reconst, ae_model = model)


            optimizer.step()


        scheduler.step()

    observer.plot_results()




def main_tensor():

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
    batch_size = 200
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    latent_dim = 10
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    encoder = SimpleLinearReluEncoder(latent_dim = latent_dim)
    decoder = SimpleLinearReluDecoder(latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    #reconstr_loss = SimpleLoss()
    reconstr_loss = MeanLpLoss(p = 2)
    #reconstr_loss = RelativeMeanLpLoss(p = 2)

    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(model.parameters(), lr = 1e-2)
    scheduler = ExponentialLR(optimizer, gamma = 0.9)


    ###--- Meta ---###
    epochs = 1
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

            print(f"{it}_{b_ind+1}/{epochs}")
            observer(loss = loss_reconst, ae_model = model)


            optimizer.step()


        scheduler.step()

    observer.plot_results()





"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    #--- main_test ---#
    #main_test_view()
    #main_test_simple()
    #main_test_lin_relu()
    main_tensor()

    pass