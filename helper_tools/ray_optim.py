
import os
import pandas as pd

from ray.tune.experiment import Trial

from pathlib import Path



"""
Ray Helper Tools
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