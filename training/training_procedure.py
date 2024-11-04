
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

"""
TrainingProcedure
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class VAEIsoTrainingProcedure(Subject):

    def __init__(
            self,
            #train_dataset: Dataset,
            train_dataloader: DataLoader,
            vae_model: VAE,
            loss, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            #batch_size: int,
            **kwargs
        ):

        super().__init__()

        self.model = vae_model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        #self.dataset = train_dataset
        self.dataloader = train_dataloader
        self.epochs = epochs
        #self.batch_size = batch_size


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            self.training_epoch(epoch = epoch)

            if self.scheduler != None:
                self.scheduler.step()


    def training_epoch(self, epoch: int):

        for iter_idx, (X_batch, _) in enumerate(self.dataloader):
            
            self.training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)

            #self.
            #print(f"{curr_epoch}_{batch_ind+1}/{self.epochs} Parameters:")
            #for param_name, value in self.model.params.items():
            #    print(f'{param_name}:\n {value.data}')


    def training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

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
            batch_loss = batch_loss, 
            infrm_dist_params = infrm_dist_params,
            model = self.model,
        )

    
    

class IsoTrainingProcedure(Subject):

    def __init__(
            self,
            #train_dataset: Dataset,
            train_dataloader: DataLoader,
            ae_model: VAE | AE,
            loss, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            #batch_size: int,
            **kwargs
        ):

        super().__init__()

        self.model = ae_model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        #self.dataset = train_dataset
        self.dataloader = train_dataloader
        self.epochs = epochs
        #self.batch_size = batch_size

        self.training_kind = 'vae' if isinstance(self.model, VAE) else 'ae'


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            self.training_epoch(epoch = epoch)

            if self.scheduler != None:
                self.scheduler.step()


    def training_epoch(self, epoch: int):

        for iter_idx, (X_batch, _) in enumerate(self.dataloader):
            
            if self.training_kind == 'vae':
                self.vae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)

            else:
                self.ae_training_iter(X_batch = X_batch, epoch = epoch, iter_idx = iter_idx)
            

            #self.
            #print(f"{curr_epoch}_{batch_ind+1}/{self.epochs} Parameters:")
            #for param_name, value in self.model.params.items():
            #    print(f'{param_name}:\n {value.data}')


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
            batch_loss = batch_loss, 
            infrm_dist_params = infrm_dist_params,
            model = self.model,
        )


    def ae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        X_hat_batch = self.model(X_batch)

        batch_loss = self.loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            batch_loss = batch_loss,
            model = self.model,
        )




class JointEpochTrainingProcedure(Subject):

    def __init__(
            self,
            #train_dataset: Dataset,
            ae_train_dataloader: DataLoader,
            regr_train_dataloader: DataLoader,
            ae_model: VAE | AE,
            regr_model: nn.Module,
            ae_loss: Loss,
            ete_loss: Loss, 
            optimizer: Optimizer,
            scheduler: LRScheduler,
            epochs: int,
            #batch_size: int,
            **kwargs
        ):

        super().__init__()

        self.ae_model = ae_model
        self.regr_model = regr_model

        self.ae_loss = ae_loss
        self.ete_loss = ete_loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        #self.dataset = train_dataset
        self.ae_dataloader = ae_train_dataloader
        self.regr_dataloader = regr_train_dataloader

        self.epochs = epochs
        #self.batch_size = batch_size

        self.training_kind = 'vae' if isinstance(self.ae_model, VAE) else 'ae'


    def __call__(self):

        epoch_progress_bar = tqdm(range(self.epochs))

        for epoch in epoch_progress_bar:

            self.training_epoch(epoch = epoch)

            if self.scheduler != None:
                self.scheduler.step()


    def training_epoch(self, epoch: int):

        for iter_idx, (X_batch, _) in enumerate(self.ae_dataloader):
            
            if self.training_kind == 'vae':
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
            batch_loss = batch_loss, 
            infrm_dist_params = infrm_dist_params,
            model = self.ae_model,
        )


    def ae_training_iter(self, X_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()
        
        _, X_hat_batch = self.ae_model(X_batch)

        batch_loss = self.ae_loss(X_batch = X_batch, X_hat_batch = X_hat_batch)

        #--- Backward Pass ---#
        batch_loss.backward()

        self.optimizer.step()

        #--- Notify Observers ---#
        self.notify_observers(
            epoch = epoch, 
            iter_idx = iter_idx, 
            batch_loss = batch_loss,
            model = self.ae_model,
        )


    def ete_training_iter(self, X_batch: torch.Tensor, y_batch: torch.Tensor, epoch: int, iter_idx: int):

        X_batch = X_batch[:, 1:]
        y_batch = y_batch[:, 1:]

        #--- Forward Pass ---#
        self.optimizer.zero_grad()

        loss_tensors = {}
        if self.training_kind == 'vae':
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
            batch_loss = batch_loss,
            model = self.regr_model,
        )





