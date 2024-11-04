
import torch
import pandas as pd
import numpy as np
import hydra

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig

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
from models import AE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE, NaiveVAESigma, NaiveVAELogSigma

from loss import (
    Loss,
    CompositeLossTerm,
    CompositeLossTermAlt,
    CompositeLossTermObs,
    Weigh,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    HuberOwn,
)

from loss.adapters_decorators import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL

from observers.ae_param_observer import AEParameterObserver
from observers import CompositeLossTermObserver, TrainingLossObserver, ModelObserver, VAELatentObserver

from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser, RobustScalingNormaliser





@hydra.main(version_base="1.2", config_path="./hydra_configs", config_name="ae_iso_cfg_sweep")
def train_AE_iso_hydra(cfg: DictConfig):

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ###--- Meta ---###
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    latent_dim = cfg.latent_dim
    n_layers_e = cfg.n_layers_e
    n_layers_d = cfg.n_layers_d

    ###--- Load Data ---###
    orig_cwd = hydra.utils.get_original_cwd()
    data_dir = Path(f"{orig_cwd}/data")
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
        #print(X_data.shape)

        X_data_isnan = X_data.isnan().all(dim = 0)
        X_data = X_data[:, ~X_data_isnan]
        #print(X_data.shape)


    ###--- Dataset---###
    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    ###--- DataLoader ---###
    dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)


    ###--- Models AE ---###
    input_dim = dataset.X_dim - 1

    # encoder = GeneralLinearReluEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_e)
    # decoder = GeneralLinearReluDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers_d)

    # model = AE(encoder = encoder, decoder = decoder)

    #--- Models NaiveVAE ---#
    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_e)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers_d)

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
    optimizer = Adam(model.parameters(), lr = cfg.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma = 0.5)


    ###--- Observation Test Setup ---###
    n_iterations = len(dataloader)
    model_obs = ModelObserver(n_epochs = epochs, n_iterations = n_iterations, model = model)
    loss_observer = TrainingLossObserver(n_epochs = epochs, n_iterations = n_iterations)


    ###--- Training Loop ---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        for iter_idx, (X_batch, _) in enumerate(dataloader):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimizer.zero_grad()
            
            X_hat_batch = model(X_batch)

            loss_reconst = reconstr_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

            #--- Backward Pass ---#
            loss_reconst.backward()

            optimizer.step()

            #--- Observer Call ---#
            model_obs(epoch = epoch, iter_idx = iter_idx, model = model)
            loss_observer(epoch = epoch, iter_idx = iter_idx, batch_loss = loss_reconst)


        scheduler.step()


    #plot_loss_tensor(observed_losses = loss_observer.losses)
    #model_obs.plot_child_param_development(child_name = 'encoder', functional = lambda t: torch.max(t) - torch.min(t))


    ###--- Test Loss ---###
    X_test = test_dataset.dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]
    X_test_hat = model(X_test)

    loss_reconst = reconstr_loss(X_batch = X_test, X_hat_batch = X_test_hat)

    results.append({
        'epochs': cfg.epochs,
        'batch_size': cfg.batch_size,
        'latent_dim': cfg.latent_dim,
        'n_layers_e': cfg.n_layers_e,
        'n_layers_d': cfg.n_layers_d,
        'learning_rate': cfg.learning_rate,
        'test_loss': loss_reconst.item(),
    })
    



"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":
    
    results = []

    train_AE_iso_hydra()

    df = pd.DataFrame(results)
    
    print(df)
    
    df.to_csv("./results/hydra_test_results.csv", index=False)