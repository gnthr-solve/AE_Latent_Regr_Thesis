
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import trange, tqdm

from observers.training_observer import Subject
from models.vae import VAE

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

    
    







