
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
from ray.tune.schedulers import ASHAScheduler

from pathlib import Path

from data_utils import DatasetBuilder, SplitSubsetFactory

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name, export_results_df 

from .config import ExperimentConfig
from .callbacks import PeriodicSaveCallback, GlobalBestModelSaver

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

logger = logging.getLogger(__name__)




def execute_tuner(current_tuner: tune.Tuner, should_raise: bool = False):
    try:
        logger.info("Starting tuner.fit()")
        results = current_tuner.fit()
        logger.info("tuner.fit() completed successfully.")
        return results
    
    except Exception as e:
        logger.error(f"An error occurred during tuning: \n{e}")
        if should_raise:
            raise e
        else:
            return None




def run_experiment_windows(
        exp_cfg: ExperimentConfig,
        save_frequency: int = 5,
        cleanup_frequency: int = 10,
        max_concurrent: int = 2,
        should_resume: bool = False,
        max_retries: int = 5,
        retry_delay: int = 20,
        replace_default_tmp: bool = False,
        restart_errored: bool = True,
    ):
    """
    Runs a Ray Tune hyperparameter optimisation experiment.
    Either begins a new experiment or resumes a previous one, if possible and desired.

    On Windows IO Permission-errors, when updating the experiment- or search-state, 
    frequently occurred and broke the experiment.
    Potential sources of the error could be parallel threads accessing the state files at the same time,
    or Antivirus software temporarily blocking access.
    This function hence attempts to resume an experiment when that exception is raised after a time delay.

    NOTE: A better working but crude solution was to change the 'save_to_dir' method of the TuneController in 
        ray.tune.execution.tune_controller
        by wrapping the state-saving calls in a (while - try-except) block and making it wait for a second between retries.
        While not recommendable, this resolved the issue and allowed executing experiments successfully.


    Args:
    ----------
        exp_cfg: ExperimentConfig
            Config passed to the trainable and containing experiment specific settings 
        save_frequency: int = 5
            Number of trials after which intermediate results are exported by PeriodicSaveCallback
        cleanup_frequency: int = 10
            Number of trials after which GlobalBestModelSaver deletes model-state checkpoints of underperforming trials.
        max_concurrent: int = 2,
            Maximum number of concurrently running trials on CPU/GPU.
        should_resume: bool = False
            Whether to resume an existing experiment or start a new one.
            (On windows experiments sometimes broke due to IO-Permission problems and could not be restarted) 
        max_retries: int = 5,
            How often to attempt to resume an experiment if it failed due to an experiment level error.
        retry_delay: int = 20,
            Time to wait between experiment-resumption attempts in seconds.
        replace_default_tmp: bool = False,
            Whether to replace the default RAY_TMPDIR
        restart_errored: bool = True,
            Whether to restart errored trials, when resuming a previous experiment.

    """

    logger.info(
        f"Experiment started\n"
        f"{'-'*80}\n"
        f"With Config: \n{exp_cfg}\n"
        f"{'-'*80}\n"
        f"And settings: \n"
        f"max_concurrent: {max_concurrent} \n"
        f"save_frequency: {save_frequency}\n"
        f"cleanup_frequency: {cleanup_frequency}\n"
        f"{'-'*80}\n"
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


    ###--- Run Config Elements ---###
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

    asha_scheduler = ASHAScheduler(
        time_attr = 'training_iteration',
        max_t = 200,
        grace_period = 15,
        reduction_factor = 2,
        brackets = 2,
    )

    checkpoint_cfg = CheckpointConfig(
        num_to_keep = 1,
        checkpoint_score_attribute = 'training_iteration', 
        checkpoint_score_order = 'max',
        # checkpoint_score_attribute = exp_cfg.optim_loss, 
        # checkpoint_score_order = exp_cfg.optim_mode
    )

    failure_cfg = FailureConfig(max_failures = 1)
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
            restart_errored = restart_errored,
        )
    
    else:
        logger.info("Setting up new experiment")
        tuner = tune.Tuner(
            trainable = tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
            tune_config = tune.TuneConfig(
                search_alg = search_alg,
                scheduler =  asha_scheduler,
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
                    restart_errored = restart_errored,
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
            f"{'-'*80}\n"
            f'Metric: \n{best_result.metrics[exp_cfg.optim_loss]}\n'
            f'With config: \n{best_result_cfg_string}\n'
            f"{'-'*80}\n\n"
        )

        # Save all results
        results_df = results.get_dataframe()
        
        export_results_df(results_df = results_df, results_dir = results_dir)




def run_experiment(
        exp_cfg: ExperimentConfig,
        save_frequency: int = 5,
        cleanup_frequency: int = 10,
        max_concurrent: int = 2,
        should_resume: bool = False,
        scheduler_kwargs: dict = {},
        replace_default_tmp: bool = False,
        restart_errored: bool = True,
    ):
    """
    Runs a Ray Tune hyperparameter optimisation experiment.
    Either begins a new experiment or resumes a previous one, if possible and desired.
    NOTE:
        On Unix systems (macOS, linux), the permission-error problem did not occur,
        hence the retry-mechanic is unnecessary.

    Args:
    ----------
        exp_cfg: ExperimentConfig
            Config passed to the trainable and containing experiment specific settings 
        save_frequency: int = 5
            Number of trials after which intermediate results are exported by PeriodicSaveCallback
        cleanup_frequency: int = 10
            Number of trials after which GlobalBestModelSaver deletes model-state checkpoints of underperforming trials.
        max_concurrent: int = 2,
            Maximum number of concurrently running trials on CPU/GPU.
        should_resume: bool = False
            Whether to resume an existing experiment or start a new one.
            (On windows experiments sometimes broke due to IO-Permission problems and could not be restarted)
        scheduler_kwargs: dict = {},
            Dictionary of kwargs for ASHA scheduler.
            Expected members:
                - 'time_attr'
                - 'max_t'
                - 'grace_period'
                - 'reduction_factor'
                - 'brackets'
            If none provided, experiment will execute without scheduling
        replace_default_tmp: bool = False
            Whether to replace the default RAY_TMPDIR
        restart_errored: bool = True
            Whether to restart errored trials, when resuming a previous experiment.
    """

    logger.info(
        f"Experiment started\n"
        f"{'-'*80}\n"
        f"With Config: \n{exp_cfg}\n"
        f"{'-'*80}\n"
        f"And settings: \n"
        f"max_concurrent: {max_concurrent} \n"
        f"save_frequency: {save_frequency}\n"
        f"cleanup_frequency: {cleanup_frequency}\n"
        f"{'-'*80}\n"
    )
    
    if replace_default_tmp:
        """
        Replace the default RAY_TMPDIR
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


    ###--- Run Config Elements ---###
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

    if scheduler_kwargs:
        asha_scheduler = ASHAScheduler(
            **scheduler_kwargs
        )
    else:
        asha_scheduler = None

    #only store latest (by training iteration) model-state checkpoint
    checkpoint_cfg = CheckpointConfig(
        num_to_keep = 1,
        checkpoint_score_attribute = 'training_iteration',
        checkpoint_score_order = 'max',
        # checkpoint_score_attribute = exp_cfg.optim_loss, 
        # checkpoint_score_order = exp_cfg.optim_mode
    )

    failure_cfg = FailureConfig(max_failures = 1)
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
            restart_errored = restart_errored,
        )
    
    else:
        logger.info("Setting up new experiment")
        tuner = tune.Tuner(
            trainable = tune.with_parameters(exp_cfg.trainable, dataset = dataset, exp_cfg = exp_cfg),
            tune_config = tune.TuneConfig(
                search_alg = search_alg,
                scheduler =  asha_scheduler,
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

    
    ###--- Run Experiment ---###    
    results = execute_tuner(tuner, should_raise = True)

    
    ###--- Export on Success ---###
    best_result = results.get_best_result()
    best_result_cfg_string = ',\n'.join([f'{param}: {value}' for param, value in best_result.config.items()])
    logger.info(
        f"Best result:\n"
        f"{'-'*80}\n"
        f'Metric: \n{best_result.metrics[exp_cfg.optim_loss]}\n'
        f'With config: \n{best_result_cfg_string}\n'
        f"{'-'*80}\n\n"
    )

    # Save all results
    results_df = results.get_dataframe()
    
    export_results_df(results_df = results_df, results_dir = results_dir)