
import os
import pandas as pd
import torch
import ray
import logging
import time

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.train import CheckpointConfig, FailureConfig, SyncConfig

from pathlib import Path

from data_utils import DatasetBuilder, SplitSubsetFactory

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name, export_results_df 

from .config import ExperimentConfig
from .callbacks import PeriodicSaveCallback, GlobalBestModelSaver

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

logger = logging.getLogger(__name__)




def execute_tuner(current_tuner: tune.Tuner):
    try:
        logger.info("Starting tuner.fit()")
        results = current_tuner.fit()
        logger.info("tuner.fit() completed successfully.")
        return results
    
    except Exception as e:
        logger.error(f"An error occurred during tuning: \n{e}")
        return None




def run_experiment(
    exp_cfg: ExperimentConfig,
    save_frequency: int = 5,
    cleanup_frequency: int = 10,
    max_concurrent: int = 2,
    should_resume: bool = False,
    max_retries: int = 20,
    retry_delay: int = 20,
    replace_default_tmp: bool = False,
    ):
    
    logger.info(
        f"Experiment started\n"
        f"---------------------------------------------------------------------------------------------\n"
        f"With Config: \n{exp_cfg}\n"
        f"---------------------------------------------------------------------------------------------\n"
        f"And settings: \n"
        f"max_concurrent: {max_concurrent} \n"
        f"save_frequency: {save_frequency}\n"
        f"cleanup_frequency: {cleanup_frequency}\n"
        f"---------------------------------------------------------------------------------------------\n"
    )

    if replace_default_tmp:
        """
        Replace the default RAY_TMPDIR on windows attempting to avoid IO-permission problems 
        """
        os.environ["RAY_TMPDIR"] = f"{str(Path.cwd().parent)}/ray_tmp"
        os.makedirs(os.environ["RAY_TMPDIR"], exist_ok = True)

    
    ###--- Experiment Meta ---###
    data_cfg = exp_cfg.data_cfg
    experiment_name = f'{exp_cfg.experiment_name}_{data_cfg.dataset_kind}_{data_cfg.normaliser_kind}'

    #--- Results Directory ---#
    results_dir = Path(f'./results/{experiment_name}/')
    os.makedirs(results_dir, exist_ok=True)

    storage_path = Path.cwd().parent / 'ray_results'
    

    ###--- Dataset Setup ---##
    normaliser = create_normaliser(data_cfg.normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_cfg.dataset_kind,
        normaliser = normaliser,
        exclude_columns = data_cfg.exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()


    ###--- Run Config ---###
    save_results_callback = PeriodicSaveCallback(
        save_frequency = save_frequency, 
        experiment_name = exp_cfg.experiment_name, 
        tracked_metrics=[exp_cfg.optim_loss, *exp_cfg.eval_metrics],
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
        checkpoint_score_attribute = 'training_iteration', 
        checkpoint_score_order = 'max',
        # checkpoint_score_attribute = exp_cfg.optim_loss, 
        # checkpoint_score_order = exp_cfg.optim_mode
    )

    failure_cfg = FailureConfig(max_failures = 5)
    sync_cfg = SyncConfig()


    ###--- Tune Config ---###
    #search_alg = BayesOptSearch()
    #search_alg = HyperOptSearch()
    search_alg = OptunaSearch()

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent = max_concurrent)


    ###--- Setup and Run Optimisation ---###
    experiment_path = str(storage_path / experiment_name)

    if tune.Tuner.can_restore(experiment_path) and should_resume:
        logger.info("Can restore and will resume")
        tuner = tune.Tuner.restore(
            path = experiment_path,
            trainable = tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
            param_space = exp_cfg.search_space,
            restart_errored = True,
        )
    
    else:
        logger.info("Setting up new experiment")
        tuner = tune.Tuner(
            trainable = tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
            tune_config = tune.TuneConfig(
                search_alg = search_alg,
                metric = exp_cfg.optim_loss,
                mode = exp_cfg.optim_mode,
                num_samples = exp_cfg.num_samples,
                trial_dirname_creator = custom_trial_dir_name,
            ),
            run_config=train.RunConfig(
                name = experiment_name,
                storage_path = storage_path,
                checkpoint_config = checkpoint_cfg,
                failure_config= failure_cfg,
                sync_config = sync_cfg,
                callbacks = [save_results_callback, global_best_model_callback],
            ),
            param_space = exp_cfg.search_space,
        )

    
    ###--- Run Experiment, retry upon Failure ---###
    attempt = 0
    results = None

    while attempt < max_retries and results is None:
        
        if attempt > 0:

            if tune.Tuner.can_restore(experiment_path):
                logger.info("Attempting to restore the tuner from the last checkpoint.")
                tuner = tune.Tuner.restore(
                    path = experiment_path,
                    trainable = tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
                    param_space = exp_cfg.search_space,
                    restart_errored = True,
                )

            else:
                logger.info("Restoring Tuner unsuccessful, aborting experiment.")
                raise PermissionError

        # Execute Tuner
        results = execute_tuner(tuner)

        if results is None:
            attempt += 1
            logger.info(f"Retrying... Attempt {attempt} of {max_retries} after {retry_delay} seconds.")
            time.sleep(retry_delay)
        else:
            logger.info("Experiment completed successfully.")
            break

    
    ###--- Export on Success or Raise on Failure---###
    if results is None:
        logger.critical("Exceeded maximum retry attempts. Experiment failed.")
        raise RuntimeError("Ray Tune experiment failed after multiple retries.")
    
    else:
        best_result = results.get_best_result()
        best_result_cfg_string = ',\n'.join([f'{param}: {value}' for param, value in best_result.config.items()])
        logger.info(
            f"Best result:\n"
            f'------------------------------------------------------------------------------------------\n'
            f'Metric: \n{best_result.metrics[exp_cfg.optim_loss]}\n'
            f'With config: \n{best_result_cfg_string}\n'
            f'------------------------------------------------------------------------------------------\n'
        )

        # Save all results
        results_df = results.get_dataframe()
        
        export_results_df(results_df = results_df, results_dir = results_dir)