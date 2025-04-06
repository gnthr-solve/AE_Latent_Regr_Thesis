
import torch

from torch import Tensor
"""
Full observations from training procedures have shape (n_epochs * n_samples, *t), where
    n_epochs: Number training of epochs.
    n_samples: Number of samples in dataset

"""

class TrainingObsConverter:
    
    def __init__(self, observations_tensor: Tensor):
        """
        Initialises Converter for observations tensor of shape (n_epochs * n_samples, *t).
        Where
            n_epochs: number of epochs.
            n_samples: size of training dataset.
            t: shape of tracked tensor.
        """
        self.observations_tensor = observations_tensor
        self.tracked_tensor_shape = observations_tensor.shape[1:]


    def to_tensor_by_epoch_split(self, n_epochs: int):
        """
        Transform observations by epoch-stacking
            (n_epochs * n_samples, *t) 
            to 
            (n_epochs, n_samples, *t).

        Args:
            n_epochs (int): Number of epochs.

        Returns:
            torch.Tensor: Transformed tensor of shape (n_epochs, n_samples, *t).
        """
        n_samples = self.observations_tensor.shape[0] // n_epochs  # Calculate dataset size based on epochs
        return self.observations_tensor.view(n_epochs, n_samples, *self.tracked_tensor_shape)


    def to_tensor_by_iteration_agg(self, n_epochs: int, n_iters: int, aggregation_fn=torch.mean):
        """
        Transform observations by epoch-stacking and batch aggregation,
            (n_epochs * n_samples, *t) 
            to 
            (n_epochs, n_iters, *t)

        Args:
            n_epochs (int): Number of epochs.
            n_iters (int): Number of iterations per epoch.
            aggregation_fn (callable): Function to aggregate samples in each batch.
                Default is torch.mean.
        
        Returns:
            torch.Tensor: Transformed tensor of shape (n_epochs, n_iters, *t).
        """
        # First reshape to (n_epochs, n_samples, *t)
        epoch_sample_tensor = self.to_tensor_by_epoch_split(n_epochs = n_epochs)
        n_samples = epoch_sample_tensor.shape[1]
        
        # Calculate batch size (samples per iteration)
        batch_size = max(1, n_samples // n_iters)  # Ensure at least 1 sample per batch
        
        result = torch.zeros(
            (n_epochs, n_iters) + tuple(self.tracked_tensor_shape), 
            dtype=self.observations_tensor.dtype, 
            device=self.observations_tensor.device,
        )
        
        for epoch in range(n_epochs):
            # Use torch.split to efficiently divide into batches
            batches = torch.split(epoch_sample_tensor[epoch], batch_size)
            
            # Process each batch, up to n_iters batches
            for i, batch in enumerate(batches):
                if i >= n_iters:  # Only process up to n_iters batches
                    break
                result[epoch, i] = aggregation_fn(batch, dim=0)
        
        return result


    def to_list_dict_by_epoch_split(self, n_epochs: int, batch_size: int = None):
        """
        Transform observations by epoch and optional batch decomposition
            (n_epochs * n_samples, *t) 
            to 
            Dict with structure {epoch_idx: [tensor_1, ..., tensor_n_samples]} - batch_size is None
            or 
            Dict with structure {epoch_idx: [batch_1, ..., batch_n_iter]} - otherwise
        
        Args:
            n_epochs (int): Number of epochs.
            batch_size (int): Size of a training batch (except last)
        """
        # First reshape to (n_epochs, n_samples, *t)
        epoch_sample_tensor = self.to_tensor_by_epoch_split(n_epochs = n_epochs)
        n_samples = epoch_sample_tensor.shape[1]
        
        if batch_size is None:
            result = {
                epoch: [epoch_sample_tensor[epoch, i] for i in range(n_samples)]
                for epoch in range(n_epochs)
            }

        else:
            result = {}
            for epoch in range(n_epochs):
                # Use torch.split for efficient tensor splitting
                batches = torch.split(epoch_sample_tensor[epoch], batch_size)
                
                # Convert to list and limit to n_iters batches
                result[epoch] = list(batches)
        
        return result


    def to_dict_by_epoch_batch_split(self, n_epochs: int, batch_size: int):
        """
        Transform observations by epoch and optional batch decomposition
            (n_epochs * n_samples, *t) 
            to 
            Dict with structure {(epoch_idx, batch_idx): batch} - otherwise
        
        Args:
            n_epochs (int): Number of epochs.
            batch_size (int): Size of a training batch (except last)
        """
        # First reshape to (n_epochs, n_samples, *t)
        epoch_sample_tensor = self.to_tensor_by_epoch_split(n_epochs = n_epochs)
        
        result = {}
        for epoch in range(n_epochs):
            # Use torch.split for efficient tensor splitting
            batches = torch.split(epoch_sample_tensor[epoch], batch_size)
            
            for batch_idx, batch in enumerate(batches):
            
                result[(epoch, batch_idx)] = batch

        return result
        
       




# batch_agg_observations = {
#     e: torch.cat([torch.mean(batch, dim = 0, keepdim = True) for batch in batches], dim = 0)
#     for e, batches in observations.items()
# }
# batch_tuple_observations = {
#     (e, i): batch
#     for e, batches in observations.items()
#     for i, batch in enumerate(batches)
# }