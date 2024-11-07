
import torch
import time
import numpy as np

from functools import wraps


"""
Timing Info Descriptor
-----------------------------------------------------------------------------------------------------------------------------------------------
This descriptor is meant to be used by the TimingDecorator below.
When the method it manages is first called the descriptor registers the name of the owner class and wraps the method
in timing logic.
While the wrapping itself is only done once and the overhead is minimal on CPU, 
on GPU the native python implementation of the descriptor can slow down execution.
This is because native python instructions are executed on CPU 
and this requires torch to interrupt GPU processing and switch to CPU execution between method calls.
"""
class MethodTimingDescriptor:

    def __init__(self, func):
        self.func = func
        self.execution_times = []


    def __get__(self, instance, owner):

        self.class_name = owner.__name__

        def wrapper(*args, **kwargs):

            start_time = time.time()
            
            result = self.func(instance, *args, **kwargs)
            
            end_time = time.time()
            exec_time = end_time - start_time
            
            self.execution_times.append(exec_time)
            
            return result
        
        return wrapper



"""
Torch based TimingDescriptor Blueprints
-----------------------------------------------------------------------------------------------------------------------------------------------
Using the time module could incur timing overhead and potentially induce context switching between GPU and CPU
-> Applying torch tracking could be beneficial 
"""
class TorchMethodTimingDescriptor:

    def __init__(self, func):
        self.func = func
        self.execution_times = []

    def __get__(self, instance, owner):
        self.class_name = owner.__name__

        def wrapper(*args, **kwargs):
            # Use PyTorch profiler to capture execution time
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                duration=1,  # Profile for 1 second
                schedule=torch.profiler.ProfilerSchedule(wait=2)  # Wait 2 seconds before next profile
            ) as prof:
                result = self.func(instance, *args, **kwargs)

            # Extract and store execution time
            for event in prof.key_activities().children():
                if event.name == self.func.__name__:
                    self.execution_times.append(event.cpu_time + event.cuda_time)

            return result

        return wrapper



class TorchMethodTimingDescriptor2:

    def __init__(self, func):
        self.func = func
        self.profile_results = None

    def __get__(self, instance, owner):
        def wrapper(*args, **kwargs):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                result = self.func(instance, *args, **kwargs)

            self.profile_results = prof.key_activities().table(sort_by="self_cpu_time", descending=True)
            return result

        return wrapper


"""
Timing Info Descriptor-Decorator
-----------------------------------------------------------------------------------------------------------------------------------------------
The __call__ method of the timing_decorator operates on a class method by instantiating a linked descriptor.
The descriptor wraps the method when it is first called to track the execution time.
The execution times are stored under the name of the class and name of the method in the _timed_methods attribute.
In this way one decorator can accumulate the execution times of different classes and their methods at the same time.
When return_average_times is called the decorator extracts the execution times of the descriptors 
that it assigned to a decorated method with __call__.
It then averages the times and returns a dictionary with all the methods it tracked as keys, and their average execution times as values.

When efficiency is an issue the decorator can easily be commented out.
"""
class TimingDecorator:

    def __init__(self):
        self._timed_methods = {}


    def __call__(self, func):

        descriptor = MethodTimingDescriptor(func)
        #descriptor = TorchMethodTimingDescriptor(func)

        self._timed_methods[id(descriptor)] = descriptor
        
        return descriptor


    def return_average_times(self):

        avg_exec_times = {
            f'{descriptor.class_name}.{descriptor.func.__name__}': np.mean(descriptor.execution_times)
            for descriptor in self._timed_methods.values()
            if descriptor.execution_times != []
        }

        return avg_exec_times


# Instantiate the decorator
timing_decorator = TimingDecorator()



