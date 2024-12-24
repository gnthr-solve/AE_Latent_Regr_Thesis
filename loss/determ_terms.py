
import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from .loss_classes import LossTerm

"""
Loss Functions - LpNorm
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LpNorm(LossTerm):

    def __init__(self, p: int):
        self.p = p


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diff = t_batch - t_hat_batch

        diff_norms: Tensor = tla.norm(diff, ord = self.p, dim = 1)

        return diff_norms
    



class RelativeLpNorm(LossTerm):

    def __init__(self, p: int):
        self.p = p


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diff = t_batch - t_hat_batch

        t_batch_norm_mean = self.norm(t_batch).mean()
        
        diff_norm = self.norm(diff) 

        return diff_norm / t_batch_norm_mean
    

    def norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = self.p, dim = 1)
        
        return batch_norms
    



class DimLpNorm(LossTerm):
    """
    LpNorm adjusted by vector dimension to mitigate losses increasing with vector dimension
    """
    def __init__(self, p: int):
        self.p = p


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diff = t_batch - t_hat_batch

        vec_dim = diff.shape[1]

        diff_norms: Tensor = tla.norm(diff, ord = self.p, dim = 1) / vec_dim

        return diff_norms
    



"""
Loss Functions - HuberLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Huber(LossTerm):

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction = 'none', delta = delta)


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        loss_batch = self.loss_fn(t_batch, t_hat_batch)

        return loss_batch.sum(dim = -1)
    


class RelativeHuber(LossTerm):

    def __init__(self, delta: float):
        
        self.loss_fn = nn.HuberLoss(reduction = 'none', delta = delta)


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        t_batch_norms = self.norm(t_batch = t_batch)

        loss_batch = self.loss_fn(t_batch, t_hat_batch) / t_batch_norms.mean()

        return loss_batch.sum(dim = -1)
    

    def norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = 2, dim = 1)
        
        return batch_norms
    



"""
Loss Functions - KMeansLoss
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class KMeansLoss(LossTerm):

    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 0.9):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.cluster_centers = torch.randn(n_clusters, latent_dim)


    def __call__(self, Z_batch: Tensor, **tensors: Tensor) -> Tensor:

        distances = torch.cdist(Z_batch, self.cluster_centers)
        min_distances, min_indices = distances.min(dim=1)

        # Update cluster centers using a running average
        for i in range(self.n_clusters):

            mask = (min_indices == i).float().unsqueeze(1)

            if mask.sum() > 0:
                new_center = (mask * Z_batch).sum(dim=0) / mask.sum()
                self.cluster_centers[i] = self.alpha * self.cluster_centers[i] + (1 - self.alpha) * new_center

        return min_distances
    



"""
Loss Functions - HuberLossOwn
-------------------------------------------------------------------------------------------------------------------------------------------
Own implementation of the Huber loss function. 
The idea was to allow conformity with the LossTerm interface, as setting reduction = 'None' preserved the complete shape.
It turned out unnecessary, as summing over the last dimension of the loss tensor solves the problem with the original Huber loss function.
"""
class HuberOwn(LossTerm):

    def __init__(self, delta: float = 1.0):
        
        self.delta = delta


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diffs = torch.abs(t_batch - t_hat_batch)

        mask = diffs > self.delta

        loss_batch = torch.zeros_like(diffs)

        # Calculate squared loss only where mask is False
        loss_batch[~mask] = 0.5 * diffs[~mask] ** 2  

        # Calculate linear loss_batch only where mask is True
        loss_batch[mask] = self.delta * (diffs[mask] - 0.5 * self.delta)


        return loss_batch.sum(dim = -1)
    
    

    
class RelativeHuberOwn(LossTerm):

    def __init__(self, delta: float = 1.0):
        
        self.delta = delta


    def __call__(self, t_batch: Tensor, t_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        diffs = torch.abs(t_batch - t_hat_batch)

        mask = diffs > self.delta

        squared_loss = 0.5 * diffs ** 2
        linear_loss = self.delta * (diffs - 0.5 * self.delta)

        loss_batch = torch.where(mask, linear_loss, squared_loss).sum(dim = -1)

        return loss_batch / self.norm(t_batch)
    

    def norm(self, t_batch: Tensor) -> Tensor:

        batch_norms: Tensor = tla.norm(t_batch, ord = 2, dim = 1)
        
        return batch_norms