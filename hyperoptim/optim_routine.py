
import os
import tempfile
import torch
import ray
import logging

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.train import Checkpoint, CheckpointConfig

from pathlib import Path

from data_utils import DatasetBuilder, SplitSubsetFactory

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Loss, Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name, PeriodicSaveCallback, GlobalBestModelSaver

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

from .config import ExperimentConfig



def run_experiment(
    exp_cfg: ExperimentConfig,
    save_frequency: int = 5,
    cleanup_frequency: int = 10,
    max_concurrent: int = 2,
):
    
    storage_path = Path.cwd().parent / 'ray_results'

    ###--- Dataset Setup ---###
    data_cfg = exp_cfg.data_cfg
    normaliser = create_normaliser(data_cfg.normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_cfg.dataset_kind,
        normaliser = normaliser,
        exclude_columns = data_cfg.exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()

    #--- Results Directory ---#
    dir_name = f'{exp_cfg.experiment_name}_{data_cfg.dataset_kind}_{data_cfg.normaliser_kind}'
    results_dir = Path(f'./results/{dir_name}/')
    os.makedirs(results_dir, exist_ok=True)


    ###--- Run Config ---###
    save_results_callback = PeriodicSaveCallback(
        save_frequency = save_frequency, 
        experiment_name = exp_cfg.experiment_name, 
        tracked_metrics=[exp_cfg.optim_loss, *exp_cfg.metrics],
        results_dir=results_dir,
    )

    global_best_model_callback = GlobalBestModelSaver(
        tracked_metric = exp_cfg.optim_loss,   
        mode = exp_cfg.optim_mode,              
        cleanup_frequency = cleanup_frequency,       
        experiment_name = exp_cfg.experiment_name,
        results_dir = results_dir,
    )

    checkpoint_cfg = CheckpointConfig(
        num_to_keep = 1, 
        checkpoint_score_attribute = exp_cfg.optim_loss, 
        checkpoint_score_order = exp_cfg.optim_mode
    )

    ###--- Tune Config ---###
    #search_alg = BayesOptSearch()
    #search_alg = HyperOptSearch()
    search_alg = OptunaSearch()

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent = max_concurrent)


    ###--- Setup and Run Optimisation ---###
    tuner = tune.Tuner(
        tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
        tune_config = tune.TuneConfig(
            search_alg = search_alg,
            metric = exp_cfg.optim_loss,
            mode = exp_cfg.optim_mode,
            num_samples = exp_cfg.num_samples,
            trial_dirname_creator = custom_trial_dir_name,
        ),
        run_config=train.RunConfig(
            name = exp_cfg.experiment_name,
            storage_path = storage_path,
            checkpoint_config = checkpoint_cfg,
            callbacks = [save_results_callback, global_best_model_callback],
        ),
        param_space = exp_cfg.search_space,
    )

    # Run the experiment
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

    # Save results
    results_df = results.get_dataframe()
    results_df.to_csv(results_dir / f'final_results.csv', index=False)