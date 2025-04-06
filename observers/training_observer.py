
import torch
import time
import pandas as pd

from torch import Tensor

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Any

from helper_tools.tensor_buffer import TensorBuffer

"""
Buffer Training Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class BufferTrainingObserver:
    """
    Observer class that tracks tensors throughout training process.
    
    Uses TensorBuffer instances to efficiently store tensors like losses,
    latent variables, gradients, or any other quantities of interest during
    ML training procedures.
    """
    
    def __init__(
        self,
        n_epochs: Optional[int] = None,
        iterations_per_epoch: Optional[int] = None,
        device: str | torch.device = "cpu",
        tensor_specs: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """
        Initialize a TrainingObserver.
        
        Args:
            n_epochs: Total number of epochs (for pre-allocation)
            iterations_per_epoch: Iterations per epoch (for pre-allocation)
            device: Device to store tensors on
            tensor_specs: Optional specifications for expected tensors 
                (each containing dtype, tensor_shape)
        """
        self.n_epochs = n_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.tensor_specs = tensor_specs or {}
        
        # Known size for pre-allocation if specified
        self.known_size = None
        if n_epochs is not None and iterations_per_epoch is not None:
            self.known_size = n_epochs * iterations_per_epoch
        
        # Create tensor buffers
        self.buffers = defaultdict(self._create_default_buffer)
        
        # Pre-initialize buffers from specs if provided
        if tensor_specs:
            for name, spec in tensor_specs.items():
                buffer_kwargs = {
                    'device': self.device,
                    'known_size': self.known_size,
                    **spec
                }
                self.buffers[name] = TensorBuffer(**buffer_kwargs)
    

    def _create_default_buffer(self):
        """
        Create a default TensorBuffer instance
        """
        return TensorBuffer(
            device=self.device,
            known_size=self.known_size
        )
    
    
    def __call__(self, **tensors: Tensor):
        """
        Record tensors for the current iteration, handling scalar values
        by adding a batch dimension. Resulting tensor will have shape (total_iterations, *tensor.shape)
        
        Args:
            **tensors: Keyword arguments mapping tensor names to tensor values
        """
        for name, tensor in tensors.items():
            tensor = tensor.detach()  # Ensure tensor is detached
            
            # Handle scalar tensors (0 dimensions) by adding a batch dimension
            if tensor.dim() == 0:
                # Convert scalar to size (1,) tensor to make it compatible with extend
                tensor = tensor.unsqueeze(0)
            
            # Use extend consistently for all tensors
            self.buffers[name].extend(tensor)
    

    def get_tensor(self, name: str):
        """
        Get the raw tensor data for a specific observation
        """
        if name not in self.buffers:
            return None
        return self.buffers[name].finalize()
    

    def get_all_tensors(self):
        """
        Get all raw tensor data as a dictionary
        """
        return {name: buffer.finalize() for name, buffer in self.buffers.items()}
    

    def to(self, device):
        """
        Move all buffers to specified device
        """
        new_device = device if isinstance(device, torch.device) else torch.device(device)
        self.device = new_device
        
        for buffer in self.buffers.values():
            buffer.to(new_device)
        
        return self




"""
TrainingObserver
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TrainingObserver:
    """
    Observer class that tracks tensors throughout training process.
    
    Uses TensorBuffer instances to efficiently store tensors like losses,
    latent variables, gradients, or any other quantities of interest during
    ML training procedures.
    """
    
    def __init__(
        self,
        n_epochs: Optional[int] = None,
        iterations_per_epoch: Optional[int] = None,
        device: str | torch.device = "cpu",
        tensor_specs: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """
        Initialize a TrainingObserver.
        
        Args:
            n_epochs: Total number of epochs (for pre-allocation)
            iterations_per_epoch: Iterations per epoch (for pre-allocation)
            device: Device to store tensors on
            tensor_specs: Optional specifications for expected tensors 
                (each containing dtype, tensor_shape)
        """
        self.n_epochs = n_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.tensor_specs = tensor_specs or {}
        
        # Known size for pre-allocation if specified
        self.known_size = None
        if n_epochs is not None and iterations_per_epoch is not None:
            self.known_size = n_epochs * iterations_per_epoch
        
        # Create tensor buffers
        self.buffers = defaultdict(self._create_default_buffer)
        
        # Pre-initialize buffers from specs if provided
        if tensor_specs:
            for name, spec in tensor_specs.items():
                buffer_kwargs = {
                    'device': self.device,
                    'known_size': self.known_size,
                    **spec
                }
                self.buffers[name] = TensorBuffer(**buffer_kwargs)


    def _create_default_buffer(self):
        """
        Create a default TensorBuffer instance
        """
        return TensorBuffer(
            device=self.device,
            known_size=self.known_size
        )
    
    
    def __call__(self, **tensors: Tensor):
        """
        Record tensors for the current iteration, handling scalar values
        by adding a batch dimension. Resulting tensor will have shape (total_iterations, *tensor.shape)
        
        Args:
            **tensors: Keyword arguments mapping tensor names to tensor values
        """
        for name, tensor in tensors.items():
            tensor = tensor.detach()  # Ensure tensor is detached
            
            # Handle scalar tensors (0 dimensions) by adding a batch dimension
            if tensor.dim() == 0:
                # Convert scalar to size (1,) tensor to make it compatible with extend
                tensor = tensor.unsqueeze(0)
            
            # Use extend consistently for all tensors
            self.buffers[name].extend(tensor)

 
    def convert_to_history(
        self, 
        tensor_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        """
        Reorganize tensors into structured chunks.
        
        Args:
            tensor_name: Optional name of specific tensor to convert.
                If None, converts all tensors.
        
        Returns:
            Dict mapping tensor names to dictionaries where:
            - Keys are split indices
            - Values are lists of tensors (one per iteration in the split)
        """
        tensor_names = [tensor_name] if tensor_name else list(self.buffers.keys())
        result = {}
        
        for name in tensor_names:
            if name not in self.buffers:
                continue
                
            buffer = self.buffers[name]
            data = buffer.finalize()
            size_data = len(data)

            history = {}
            
            # Skip empty buffers
            if size_data == 0 or data is None:
                result[name] = {}
                continue
            
            split_size = int(size_data / epochs)
            # Use torch.split to efficiently split the tensor
            # This returns a tuple of tensors, each containing split_size elements
            # (except possibly the last one which might be smaller)
            epoch_chunks = torch.split(data, split_size)
            
            # Convert each chunk into a list and add to history
            for i, chunk in enumerate(epoch_chunks):

                if batch_size is not None:
                    batches = torch.split(chunk, batch_size)
                    history[i] = [batch for batch in batches]

                else:
                    history[i] = chunk
            
            result[name] = history
            
        return result if tensor_name is None else result[tensor_name]
    
    

    def get_tensor(self, name: str):
        """
        Get the raw tensor data for a specific observation
        """
        if name not in self.buffers:
            return None
        return self.buffers[name].finalize()
    

    def get_all_tensors(self):
        """
        Get all raw tensor data as a dictionary
        """
        return {name: buffer.finalize() for name, buffer in self.buffers.items()}
    

    def to(self, device):
        """
        Move all buffers to specified device
        """
        new_device = device if isinstance(device, torch.device) else torch.device(device)
        self.device = new_device
        
        for buffer in self.buffers.values():
            buffer.to(new_device)
        
        return self


