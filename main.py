
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from hydra import initialize, compose
from hydra.utils import instantiate
from pathlib import Path
from tqdm import tqdm

from datasets import TensorDataset
from models import SimpleEncoder, SimpleDecoder, SimpleAutoencoder, SimpleLoss

from helper_tools import get_valid_batch_size


"""
Main Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def main_test_view():
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR

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

    dataset = TensorDataset(joint_data_df)
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

    encoder = SimpleEncoder(input_dim = dataset.X_dim, latent_dim = latent_dim)
    decoder = SimpleDecoder(input_dim = dataset.X_dim, latent_dim = latent_dim)

    autoencoder = SimpleAutoencoder(encoder = encoder, decoder = decoder)


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




def main_test():
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ExponentialLR

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)


    ###--- Load Data ---###
    data_dir = Path("./data")
    joint_data_df = pd.read_csv(data_dir / "data_joint.csv")


    ###--- Dataset & DataLoader ---###
    batch_size = 32

    dataset = TensorDataset(joint_data_df)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    

    ###--- Models ---###
    latent_dim = 10

    encoder = SimpleEncoder(input_dim = dataset.X_dim, latent_dim = latent_dim)
    decoder = SimpleDecoder(input_dim = dataset.X_dim, latent_dim = latent_dim)

    autoencoder = SimpleAutoencoder(encoder = encoder, decoder = decoder)

    loss = SimpleLoss()


    ###--- Optimizer & Scheduler ---###
    optimizer = Adam(autoencoder.parameters(), lr = 1e-3)
    scheduler = ExponentialLR(optimizer, gamma = 0.99)


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
                #f"Shape y_batch: \n {y_batch.shape}\n"
                #f"y_batch: \n {y_batch}\n"
            )

            #--- Forward Pass ---#
            optimizer.zero_grad()

            X_hat_batch = autoencoder(X_batch)

            loss_reconst = loss(X_batch, X_hat_batch)
            print(f"Reconstruction Loss: {loss_reconst}")


            #--- Backward Pass ---#
            loss_reconst.backward()

            optimizer.step()


        scheduler.step()





"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    #--- main_test ---#
    main_test()