
###--- Libraries ---###
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt

from pathlib import Path
from torch import Tensor


"""
Results - Read and retrieve experiment results for connected experiments
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def read_experiment_results(experiment_names: list[str], results_dir: Path = None) -> dict[str, pd.DataFrame]:
    """
    Reads the result csv files for several experiment subdirectories contained in "results_dir" separately.

    Parameters
    ----------
        experiment_names: list[str]
            List of experiment names, determining the subdirectories in which to find the respective results.
        results_dir: Path = None
            Path to results directory containing the experiment results.
            If not specified assume the directory is './results/'

    Returns:
        experiment_results: dict[str, pd.DataFrame]
            Dictionary with experiment names as keys and hyperparameter optimisation results dataframe as values.
    """
    results_dir = results_dir or Path('./results')

    experiment_results: dict[str, pd.DataFrame] = {}
    for experiment_name in experiment_names:

        experiment_dir = results_dir / experiment_name

        experiment_results[experiment_name] = pd.read_csv(experiment_dir / 'final_results.csv', low_memory = False)

    return experiment_results



"""
Results - Collection of regression metrics
-------------------------------------------------------------------------------------------------------------------------------------------
Direct calculation of standard regression metrics, per dimension and overall, 
that does not require Evaluation instances
"""
def regression_metrics(y_true: Tensor, y_pred: Tensor) -> dict[str, Tensor]:
    """
    Calculates common regression evaluation metrics, both per dimension and averaged by dimension.
    Included metrics:
        - MSE
        - RMSE
        - MAE
        - R-squared

    Parameters
    ----------
        y_true: Tensor
            Tensor of actual regression target labels.
        y_pred: Tensor
            Tensor of target labels predicted by a model.

    Returns:
        metrics: dict[str, Tensor]
            Dictionary with metric names as keys and result tensors as values.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Per-dimension errors
    errors = y_true - y_pred
    
    #--- MSE (per dimension and combined) ---#
    mse_per_dim = torch.mean(errors**2, dim=0)
    mse_combined = torch.mean(errors**2)
    
    #--- RMSE ---#
    rmse_per_dim = torch.sqrt(mse_per_dim)
    rmse_combined = torch.sqrt(mse_combined)
    
    #--- MAE ---#
    mae_per_dim = torch.mean(torch.abs(errors), dim = 0)
    mae_combined = torch.mean(torch.abs(errors))
    
    #--- R-squared (per dimension) ---#
    ss_res = torch.sum(errors**2, dim = 0)
    ss_tot = torch.sum((y_true - y_true.mean(dim = 0))**2, dim = 0)
    
    r2_per_dim = 1 - (ss_res / ss_tot)
    
    #--- Combined R-squared (treats both dimensions as independent observations) ---#
    ss_res_combined = torch.sum(errors**2)
    ss_tot_combined = torch.sum((y_true - y_true.mean(dim = 0))**2)
    r2_combined = 1 - (ss_res_combined / ss_tot_combined)
    
    return {
        'mse_dim': mse_per_dim,
        'mse': mse_combined,
        'rmse_dim': rmse_per_dim,
        'rmse': rmse_combined,
        'mae_dim': mae_per_dim,
        'mae': mae_combined,
        'r2_dim': r2_per_dim,
        'r2': r2_combined
    }
