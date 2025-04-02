
import torch

from torch import Tensor, dtype, device

from collections import defaultdict
from typing import Callable, Optional, Sequence


"""
TensorBuffer
-------------------------------------------------------------------------------------------------------------------------------------------
An efficient, flexible tensor container for ML training observation.
    
Features:
    - Memory efficient with minimal allocations
    - Adaptive sizing with configurable growth strategy
    - Support for both known and unknown total sizes
    - Optimized for GPU usage with device management
    - Vectorised operations on the entire buffer
"""

class TensorBuffer:
    
    def __init__(
        self, 
        tensor_shape: int | tuple[int] | Sequence[int] | None = None,
        dtype: dtype | None = None,
        device: str | device | None = None,
        initial_capacity: int = 1024,
        growth_factor: int | float = 2.0,       
        known_size: int | None = None,
    ):
        """
        Initialise a TensorBuffer.
        
        Args:
            tensor_shape: Shared shape of the stored tensors
            dtype: Shared dtype of the stored tensors
            device: Device to store tensors on
            initial_capacity: Initial size of the buffer
            growth_factor: Factor by which capacity is increased if exhausted
            known_size: If given, preallocates memory for known capacity
        """
        self.tensor_shape = tensor_shape # Shape of individual tensors (excluding batch dim)
        self.dtype = dtype # PyTorch data type 
        self.device = device # PyTorch device
        self.growth_factor = growth_factor # How much to expand when needed
        
        # Internal storage state
        self._idx = 0
        self._data = None
        self._initialized = False
        
        # Set capacity based on known size or initial capacity
        self._capacity = known_size if known_size is not None else initial_capacity
            
        # Pre-allocate if we have complete information
        if tensor_shape is not None and dtype is not None and device is not None:
            self._initialize_storage()
    

    def _initialize_storage(self):
        """
        Initialize the storage tensor with proper shape
        """
        if not isinstance(self.tensor_shape, tuple):
            self.tensor_shape = tuple(self.tensor_shape) if hasattr(self.tensor_shape, '__iter__') else (self.tensor_shape,)
            
        buffer_shape = (self._capacity,) + self.tensor_shape
        self._data = torch.zeros(buffer_shape, dtype=self.dtype, device=self.device)
        self._initialized = True
    

    def _resize(self, min_capacity: int):
        """
        Resize internal buffer to accommodate at least min_capacity elements
        """
        new_capacity = max(min_capacity, int(self._capacity * self.growth_factor))
        
        buffer_shape = (new_capacity,) + self.tensor_shape
        new_buffer = torch.zeros(buffer_shape, dtype=self.dtype, device=self.device)
        new_buffer[:self._idx] = self._data[:self._idx]
        self._data = new_buffer
        self._capacity = new_capacity
    

    def append(self, tensor: Tensor):
        """
        Add a single tensor to the buffer
        """
        tensor = tensor.detach()  # Detach to avoid tracking computation graph
        
        # Initialize if first tensor
        if not self._initialized:
            self.dtype = tensor.dtype if self.dtype is None else self.dtype
            self.device = tensor.device if self.device is None else self.device
            
            if self.tensor_shape is None:
                self.tensor_shape = tensor.shape
                
            self._initialize_storage()
            
        # Validate tensor shape
        if tensor.shape != self.tensor_shape:
            raise ValueError(f"Expected tensor shape {self.tensor_shape}, got {tensor.shape}")
        
        # Resize if needed
        if self._idx >= self._capacity:
            self._resize(self._capacity * 2)
            
        # Store the tensor
        self._data[self._idx] = tensor
        self._idx += 1
    

    def extend(self, tensors: Tensor | Sequence[Tensor]):
        """
        Add multiple tensors at once (optimized for batches)
        """
        # Handle tensor batch (first dimension is batch size)
        if isinstance(tensors, torch.Tensor) and len(tensors.shape) > 0:
            batch_tensor = tensors.detach()
            batch_size = batch_tensor.shape[0]
            
            # Initialize if needed
            if not self._initialized:
                self.dtype = batch_tensor.dtype if self.dtype is None else self.dtype
                self.device = batch_tensor.device if self.device is None else self.device
                
                if self.tensor_shape is None:
                    # Extract shape without batch dimension
                    self.tensor_shape = batch_tensor.shape[1:] if len(batch_tensor.shape) > 1 else ()
                
                self._initialize_storage()
            
            # Check if shapes align (excluding batch dimension)
            expected_batch_shape = (batch_size,) + self.tensor_shape
            if batch_tensor.shape != expected_batch_shape:
                raise ValueError(f"Expected batch tensor shape {expected_batch_shape}, got {batch_tensor.shape}")
                
            # Ensure we have capacity
            if self._idx + batch_size > self._capacity:
                self._resize(self._idx + batch_size)
                
            # Add all tensors from batch at once (vectorized operation)
            self._data[self._idx:self._idx+batch_size] = batch_tensor
            self._idx += batch_size

        else:
            # Handle iterable of tensors
            for tensor in tensors:
                self.append(tensor)
    

    def get(self, idx = None):
        """
        Get tensor(s) at specified index/indices
        """
        if not self._initialized:
            return None
            
        if idx is None:
            return self._data[:self._idx]
        
        return self._data[idx]
    

    def finalize(self):
        """
        Return a tensor containing only the used elements
        """
        if not self._initialized:
            return None
        return self._data[:self._idx]
    

    def __len__(self):
        """
        Return count of stored tensors
        """
        return self._idx
    

    def __getitem__(self, idx):
        """
        Support indexing with [] syntax
        """
        return self.get(idx)
    

    def clear(self):
        """
        Reset the buffer without deallocating memory
        """
        self._idx = 0
    

    @property
    def shape(self):
        """
        Return shape of current data
        """
        if not self._initialized:
            return None
        return (self._idx,) + self.tensor_shape
    

    def to(self, device):
        """
        Move buffer to specified device
        """
        if not self._initialized:
            self.device = device
            return self
        
        self._data = self._data.to(device)
        self.device = device
        return self
    
