
import os
import shutil
import pandas as pd
import numpy as np
import glob
import logging
import threading

import ray.tune as tune

from ray.tune import Callback
from ray.tune.experiment import Trial

from pathlib import Path
from collections import deque


logger = logging.getLogger(__name__)

"""
Ray Callbacks - PeriodicSaveCallback
-------------------------------------------------------------------------------------------------------------------------------------------
Saves the results of all previous trials as a pandas dataframe.
"""
class PeriodicSaveCallback(Callback):
    """
    tune.Callback subclass to periodically save Trial results to csv file.
    """
    def __init__(self, save_frequency: int, experiment_name: str, tracked_metrics: list[str], results_dir: Path):
        """
        Init method.

        Args:
        --------
            save_frequency: int
                Number of terminated trials after which to save intermediate results.
            experiment_name: str
                Name of experiment, used for csv file name. 
            tracked_metrics: list[str]
                Names of (reported) metrics to include in intermediate results.
            results_dir: Path
                Directory where to save csv.
        """
        self.experiment_name = experiment_name
        self.tracked_metrics = tracked_metrics
        
        self.save_frequency = save_frequency
        self.trial_counter = 0
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)


    def on_trial_complete(self, iteration, trials: list[Trial], trial: Trial, **info):
        """
        Concrete implementation of template method.
        Increments a counter for completed trials. 
        If counter is multiple of save_frequency:
            - retrieves results of previously terminated or errored trials (avoid incomplete trials)
            - exports results as csv 
        """
        self.trial_counter += 1

        if self.trial_counter % self.save_frequency == 0:
            
            results = []
            for t in trials:

                # 'trials' argument corresponds to all trials, including currently running ones --> limit to completed
                if t.status in [t.TERMINATED, t.ERROR]:
                    result = t.last_result.copy()
                    result_data = {metric: result.get(metric, None) for metric in self.tracked_metrics}
                    result_data.update(result.get('config', {}))

                    results.append(result_data)

            results_df = pd.DataFrame(results)

            # Save the DataFrame to CSV
            results_df.to_csv(self.results_dir / f'{self.experiment_name}_interim.csv', index=False)

            logger.info(f"Saved interim results after {self.trial_counter} trials.")




"""
Ray Callbacks - GlobalBestModelSaver
-------------------------------------------------------------------------------------------------------------------------------------------
Transfers the currently best models from the checkpoint directory to the results directory.
Deletes model checkpoints from previous trials at a determined frequency.
"""
class GlobalBestModelSaver(Callback):
    """
    tune.Callback subclass to store the state-dict of currently best performing model (or models for e.g. composite)
    in the experiment directory.
    """
    def __init__(self, tracked_metric: str, mode: str, cleanup_frequency: int, results_dir: Path):
        """
        Init method.

        Args:
        --------
            tracked_metric: str
                Name of optimised metric by which to evaluate/compare model performance.
            mode: str
                Optimisation mode ('min' or 'max').
            cleanup_frequency: int
                Number of completed trials after which to remove model checkpoints from Tune Trial directory,
                to save on memory (model-state can be large and quickly add up).
            results_dir: Path
                Experiment directory where to store best performing model checkpoint.
        """
        
        self.tracked_metric = tracked_metric
        self.mode = mode 
        self.cleanup_frequency = cleanup_frequency
        self.trial_counter = 0
        
        self.global_best_metric = None
        self.global_best_checkpoint_paths = []
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
    

    def _is_better(self, current, best):

        if self.mode == "min":
            return current < best
        
        else:
            return current > best
    

    def on_trial_complete(self, iteration, trials: list[Trial], trial: Trial, **info):
        """
        Concrete implementation of template method.
        Increments a counter for completed trials.

        If the performance of the Trial, that reports completion, is better than previously seen Trials,
        method moves all '.pt' files from Tune Trial Checkpoint directory to experiment directory.
        
        If counter is a multiple of cleanup_frequency, all checkpoints are deleted from Tune Trial directories.
        """
        self.trial_counter += 1
        
        last_result = trial.last_result

        if self.tracked_metric in last_result:

            trial_metric = last_result[self.tracked_metric]
            if self.global_best_metric is None or self._is_better(trial_metric, self.global_best_metric):
                # Update the global best metric
                self.global_best_metric = trial_metric

                # Delete the previous global best model checkpoints if they exist
                for path in self.global_best_checkpoint_paths:
                    if os.path.exists(path):
                        os.remove(path)

                self.global_best_checkpoint_paths = []

                # Save the new global best model checkpoints
                checkpoint = trial.checkpoint
                if checkpoint:

                    with checkpoint.as_directory() as checkpoint_dir:

                        model_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

                        for src_model_path in model_files:
                            model_file_name = os.path.basename(src_model_path)
                            dest_model_path = os.path.join(self.results_dir, model_file_name)

                            logger.debug(f"Attempting to copy {src_model_path} to {dest_model_path}")
                            shutil.copyfile(src_model_path, dest_model_path)

                            self.global_best_checkpoint_paths.append(dest_model_path)
                            logger.info(f"Saved {model_file_name} to {dest_model_path}")

                logger.info(f"New global best model with {self.tracked_metric}: {self.global_best_metric}")

        # Periodically clean up old checkpoints
        if self.trial_counter % self.cleanup_frequency == 0:
            self._cleanup_checkpoints(trials, exclude_trial=trial)
    

    def _cleanup_checkpoints(self, trials: list[Trial], exclude_trial: Trial):
        """
        Removes old trial checkpoints from completed trials (terminated or errored) for reduced memory usage.
        
        Args:
        --------
            trials: list[Trial]
                List of all currently registered Trial's.
            exclude_trial: Trial
                Trial to be excluded from deletion that just reported completion.
        """
        for t in trials:

            if t.status not in [t.TERMINATED, t.ERROR]:
                continue

            if t is not exclude_trial and t.checkpoint:

                checkpoint = t.checkpoint

                if checkpoint:
                    # Remove the checkpoint directory
                    checkpoint_path = checkpoint.path  # Get the path to the checkpoint
                    if os.path.exists(checkpoint_path):
                        try:
                            # Attempt to delete the checkpoint directory
                            logger.debug(f"Attempting to delete directory at {checkpoint_path}")
                            shutil.rmtree(checkpoint_path, ignore_errors=True)
                        except Exception as e:
                            logger.error(f"Error deleting checkpoint {checkpoint_path}: {e}")

        logger.info(f"Removed old checkpoints after {self.trial_counter} trials.")


