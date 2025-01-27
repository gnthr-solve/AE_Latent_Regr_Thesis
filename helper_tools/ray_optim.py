
import os
import tempfile
import torch
import pandas as pd
import torch.nn as nn

from pathlib import Path
from typing import Callable

from torch import Tensor

from ray import train
from ray.train import Checkpoint
from ray.tune.experiment import Trial

from .torch_grad import no_grad_decorator



"""
Ray Helper Tools - Loss Reporter
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RayTuneLossReporter:

    def __init__(self, checkpoint_condition: Callable[[int], bool]):
        self.checkpoint_condition = checkpoint_condition
        self.current_losses: dict[str, float] = {}
        

    @no_grad_decorator
    def observe_loss(self, name: str, loss_batch: Tensor):
        """
        Callback method for CompositeLossTerm
        """
        self.current_losses[name] = loss_batch.mean().item()
    

    def report(self, epoch: int, **models: nn.Module):
        """
        Reports losses and optionally creates checkpoint
        """
        if self.checkpoint_condition(epoch):

            with tempfile.TemporaryDirectory() as tmp_dir:

                for name, model in models.items():

                    torch.save(
                        model.state_dict(), 
                        os.path.join(tmp_dir, f"{name}.pt")
                    )

                checkpoint = Checkpoint.from_directory(tmp_dir)

                train.report(self.current_losses, checkpoint = checkpoint)
                print(f'Checkpoint created at epoch {epoch + 1}')

        else:
            train.report(self.current_losses)




"""
Ray Helper Tools - Export results
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def export_results_df(results_df: pd.DataFrame, results_dir: Path):
    """
    Export results of concluded Ray Tune experiment to results directory of experiment.

    Parameters
    ----------
        results_df: pd.DataFrame
            DataFrame created by calling .get_dataframe() on the ResultGrid returned by Tuner.fit().
        results_dir: Path
            Path of experiment directory to export the results dataframe to.
    """

    #--- Remove 'config/' prefix of hyperparameters ---#
    export_df = results_df.rename(columns=lambda col: col.replace('config/', '') if col.startswith('config/') else col)

    #--- Drop columns containing only Tune meta information ---#
    drop_cols = [
        'checkpoint_dir_name',
        'done',
        'training_iteration',
        'trial_id',
        'date',
        'time_this_iter_s',
        'pid',
        'hostname',
        'node_ip',
        'time_since_restore',
        'iterations_since_restore',
        'should_checkpoint',
        'logdir'
    ]

    export_df.drop(columns = drop_cols, inplace = True)

    #--- Export ---#
    export_df.to_csv(results_dir / f'final_results.csv', index=False)




"""
Ray Trial-Name Creator
-------------------------------------------------------------------------------------------------------------------------------------------
Creates a shorter name for the trials and their directories to avoid an exception on windows where the path is too long,
due to too many search parameters (name usually is concatenation of key:value and date).
"""
def custom_trial_dir_name(trial: Trial):
    """
    Create Trial directory name based on Trial ID. Used internally by Ray Tune.

    Parameters
    ----------
        trial: experiment.Trial
            Trial whose directory is to be named.
        
    Returns:
        str
            Trial-directory name string.
    """
    return f'trial_{trial.trial_id}'