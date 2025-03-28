
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch import nn

from tqdm import trange, tqdm

from observers.iter_observer import Subject

from models.vae import VAE
from models.autoencoders import AE

from loss import Loss

from helper_tools import AbortTrainingError

"""
TrainingProcedure - Joint Epochs
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class JointEpochTrainingProcedure(Subject):

    def __init__(
            self,
            ae_train_dataloader: DataLoader,
            regr_train_dataloader: DataLoader,
            ae_model: VAE | AE,
            regr_model: nn.Module,
            ae_loss: Loss,
            ete_loss: Loss, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            **kwargs
        ):

        super().__init__()

        self.ae_model = ae_model
        self.regr_model = regr_model

        self.ae_loss = ae_loss
        self.ete_loss = ete_loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.ae_dataloader = ae_train_dataloader
        self.regr_dataloader = regr_train_dataloader

        self.epochs = epochs

        self.ae_training_kind = 'vae' if isinstance(self.ae_model, VAE) else 'ae'


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

        for iter_idx, (X_batch, _) in enumerate(self.ae_dataloader):
            
            if self.ae_training_kind == 'vae':
                self.vae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)
                
            else:
                self.ae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)
            

        for iter_idx, (X_batch, y_batch) in enumerate(self.regr_dataloader):

            self.ete_training_iter(X_batch = X_batch, y_batch = y_batch, epoch = epoch, iter_idx = iter_idx)


    def vae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        Z_batch, infrm_dist_params, genm_dist_params = self.ae_model(X_batch)

        batch_loss = self.ae_loss(
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
            Z_batch = Z_batch,
            infrm_dist_params = infrm_dist_params,
            model = self.ae_model,
        )


    def ae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        Z_batch, X_hat_batch = self.ae_model(X_batch)

        batch_loss = self.ae_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            Z_batch = Z_batch,
            model = self.ae_model,
        )


    def ete_training_iter(self, X_batch: torch.Tensor, y_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]
        y_batch = y_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()

        loss_tensors = {}
        if self.ae_training_kind == 'vae':
            Z_batch, infrm_dist_params, genm_dist_params = self.ae_model(X_batch)
            loss_tensors['Z_batch'] = Z_batch
            loss_tensors['infrm_dist_params'] = infrm_dist_params
            loss_tensors['genm_dist_params'] = genm_dist_params
        
        else:
            Z_batch, X_hat_batch = self.ae_model(X_batch)
            loss_tensors['Z_batch'] = Z_batch
            loss_tensors['X_hat_batch'] = X_hat_batch

        y_hat_batch = self.regr_model(Z_batch)

        batch_loss = self.ete_loss(
            X_batch = X_batch,
            y_batch = y_batch,
            y_hat_batch = y_hat_batch,
            **loss_tensors
        )

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx,
            Z_batch = Z_batch,
            model = self.regr_model,
        )




"""
TrainingProcedure - Sequential Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class JointSequentialTrainingProcedure(Subject):

    def __init__(
            self,
            ae_train_dataloader: DataLoader,
            regr_train_dataloader: DataLoader,
            ae_model: VAE | AE,
            regr_model: nn.Module,
            ae_loss: Loss,
            ete_loss: Loss,
            ae_optimiser: Optimizer,
            regr_optimiser: Optimizer,
            ae_scheduler: LRScheduler,
            regr_scheduler: LRScheduler,
            epochs: int,
            **kwargs
        ):

        super().__init__()

        self.ae_model = ae_model
        self.regr_model = regr_model

        self.ae_loss = ae_loss
        self.ete_loss = ete_loss

        self.ae_optimiser = ae_optimiser
        self.regr_optimiser = regr_optimiser

        self.ae_scheduler = ae_scheduler
        self.regr_scheduler = regr_scheduler

        self.ae_dataloader = ae_train_dataloader
        self.regr_dataloader = regr_train_dataloader

        self.epochs = epochs

        self.ae_training_kind = 'vae' if isinstance(self.ae_model, VAE) else 'ae'


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            try:
                self.training_epoch_ae(epoch = epoch)
                
            except AbortTrainingError:
                return
            
            if self.ae_scheduler != None:
                self.ae_scheduler.step()
        

        for epoch in epoch_progress_bar:

            try:
                self.training_epoch_regr(epoch = epoch)
                
            except AbortTrainingError:
                return
            
            if self.regr_scheduler != None:
                self.regr_scheduler.step()


    def training_epoch_ae(self, epoch: int):

        for iter_idx, (X_batch, _) in enumerate(self.ae_dataloader):
            
            if self.ae_training_kind == 'vae':
                self.vae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)
                
            else:
                self.ae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)


    def training_epoch_regr(self, epoch: int):

        for iter_idx, (X_batch, y_batch) in enumerate(self.regr_dataloader):

            self.ete_training_iter(X_batch = X_batch, y_batch = y_batch, epoch = epoch, iter_idx = iter_idx)


    def vae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.ae_optimiser.zero_grad()
        
        Z_batch, infrm_dist_params, genm_dist_params = self.ae_model(X_batch)

        batch_loss = self.ae_loss(
            X_batch = X_batch,
            Z_batch = Z_batch,
            genm_dist_params = genm_dist_params,
            infrm_dist_params = infrm_dist_params,
        )

        #--- Backward Pass ---#
        batch_loss.backward()

        self.ae_optimiser.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            Z_batch = Z_batch,
            infrm_dist_params = infrm_dist_params,
            model = self.ae_model,
        )


    def ae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.ae_optimiser.zero_grad()
        
        Z_batch, X_hat_batch = self.ae_model(X_batch)

        batch_loss = self.ae_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

        #--- Backward Pass ---#
        batch_loss.backward()

        self.ae_optimiser.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            Z_batch = Z_batch,
            model = self.ae_model,
        )


    def ete_training_iter(self, X_batch: torch.Tensor, y_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]
        y_batch = y_batch[:, 1:]

        #--- Forward Pass ---#
        self.regr_optimiser.zero_grad()

        loss_tensors = {}
        if self.ae_training_kind == 'vae':
            Z_batch, infrm_dist_params, genm_dist_params = self.ae_model(X_batch)
            loss_tensors['Z_batch'] = Z_batch
            loss_tensors['infrm_dist_params'] = infrm_dist_params
            loss_tensors['genm_dist_params'] = genm_dist_params
        
        else:
            Z_batch, X_hat_batch = self.ae_model(X_batch)
            loss_tensors['Z_batch'] = Z_batch
            loss_tensors['X_hat_batch'] = X_hat_batch

        y_hat_batch = self.regr_model(Z_batch)

        batch_loss = self.ete_loss(
            X_batch = X_batch,
            y_batch = y_batch,
            y_hat_batch = y_hat_batch,
            **loss_tensors
        )

        #--- Backward Pass ---#
        batch_loss.backward()

        self.regr_optimiser.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx,
            Z_batch = Z_batch,
            model = self.regr_model,
        )



