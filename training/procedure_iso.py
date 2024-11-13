
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch import nn

from tqdm import trange, tqdm

from observers.training_observer import Subject

from models.vae import VAE
from models.autoencoders import AE

from loss import Loss

from helper_tools import AbortTrainingError


"""
TrainingProcedure - AE | VAE Iso
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AEIsoTrainingProcedure(Subject):

    def __init__(
            self,
            train_dataloader: DataLoader,
            ae_model: VAE | AE,
            loss, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            **kwargs
        ):

        super().__init__()

        self.model = ae_model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataloader = train_dataloader
        self.epochs = epochs

        self.ae_training_kind = 'vae' if isinstance(self.model, VAE) else 'ae'


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            try:
                self.training_epoch(epoch = epoch)
                
            except AbortTrainingError:
                return

            if self.scheduler != None:
                self.scheduler.step()


    def training_epoch(self, epoch: int):

        for iter_idx, (X_batch, _) in enumerate(self.dataloader):
            
            if self.ae_training_kind == 'vae':
                self.vae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)

            else:
                self.ae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)


    def vae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        Z_batch, infrm_dist_params, genm_dist_params = self.model(X_batch)

        batch_loss = self.loss(
            X_batch = X_batch,
            Z_batch = Z_batch,
            genm_dist_params = genm_dist_params,
            infrm_dist_params = infrm_dist_params,
        )

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            infrm_dist_params = infrm_dist_params,
            model = self.model,
        )


    def ae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        Z_batch, X_hat_batch = self.model(X_batch)

        batch_loss = self.loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx,
            model = self.model,
        )



