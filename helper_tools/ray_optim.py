
import os
import pandas as pd

from ray.tune.experiment import Trial

from pathlib import Path



def export_results_df(results_df: pd.DataFrame, results_dir: Path):

    export_df = results_df.rename(columns=lambda col: col.replace('config/', '') if col.startswith('config/') else col)

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

    export_df.to_csv(results_dir / f'final_results.csv', index=False)

"""
Ray Trial-Name Creator
-------------------------------------------------------------------------------------------------------------------------------------------
Creates a shorter name for the trials and their directories to avoid an exception on windows where the path is too long,
due to too many search parameters (name usually is concatenation of key:value and date).
"""
def custom_trial_dir_name(trial: Trial):
    return f'trial_{trial.trial_id}'