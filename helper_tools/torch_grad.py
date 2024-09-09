
import torch
import torch.linalg as tla

from functools import wraps
from torch.distributions import MultivariateNormal


"""
No-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
A wrapper decorator that enables the torch.no_grad() context to avoid gradient tracking.
"""
def no_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper


"""
No-Grad Descriptor
-------------------------------------------------------------------------------------------------------------------------------------------
A descriptor class that enables the torch.no_grad() context to avoid gradient tracking.
"""
class NoGradDescriptor:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):

        def wrapper(*args, **kwargs):

            with torch.no_grad():
                result = self.func(instance, *args, **kwargs)

            return result
        
        return wrapper
    
"""
Enable-Grad Decorator
-------------------------------------------------------------------------------------------------------------------------------------------
Wraps a function or method in torch.enable_grad context manager.
"""
def enable_grad_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper
