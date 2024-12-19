
import os
import shutil
import pandas as pd
import glob

import ray.tune as tune

from ray.tune import Callback
from ray.tune.experiment import Trial

from pathlib import Path


"""
Ray Callbacks - PeriodicSaveCallback
-------------------------------------------------------------------------------------------------------------------------------------------
Saves the results of all previous trials as a pandas dataframe.
"""
class PeriodicSaveCallback(Callback):

    def __init__(self, save_frequency, experiment_name, tracked_metrics: list[str], results_dir):
        self.experiment_name = experiment_name
        self.tracked_metrics = tracked_metrics
        
        self.save_frequency = save_frequency
        self.trial_counter = 0
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)


    def on_trial_complete(self, iteration, trials: list[Trial], trial: Trial, **info):

        self.trial_counter += 1

        if self.trial_counter % self.save_frequency == 0:
            
            results = []
            for t in trials:

                if t.status in [t.TERMINATED, t.ERROR]:
                    result = t.last_result.copy()
                    #print(result)
                    result_data = {metric: result.get(metric, None) for metric in self.tracked_metrics}
                    result_data.update(result.get('config', {}))

                    results.append(result_data)

            results_df = pd.DataFrame(results)

            # Save the DataFrame to CSV
            results_df.to_csv(self.results_dir / f'{self.experiment_name}_interim.csv', index=False)

            print(f"Saved interim results after {self.trial_counter} trials.")




"""
Ray Callbacks - GlobalBestModelSaver
-------------------------------------------------------------------------------------------------------------------------------------------
Transfers the currently best models from the checkpoint directory to the results directory.
Deletes model checkpoints from previous trials at a determined frequency.
"""
class GlobalBestModelSaver(Callback):

    def __init__(self, tracked_metric, mode, cleanup_frequency, experiment_name, results_dir):
        self.experiment_name = experiment_name
        
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

                            shutil.copyfile(src_model_path, dest_model_path)

                            self.global_best_checkpoint_paths.append(dest_model_path)
                            print(f"Saved {model_file_name} to {dest_model_path}")

                print(f"New global best model with {self.tracked_metric}: {self.global_best_metric}")

        # Periodically clean up old checkpoints
        if self.trial_counter % self.cleanup_frequency == 0:
            self._cleanup_checkpoints(trials, exclude_trial=trial)
    

    def _cleanup_checkpoints(self, trials: list[Trial], exclude_trial: Trial):
        
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
                            shutil.rmtree(checkpoint_path, ignore_errors=True)
                        except Exception as e:
                            print(f"Error deleting checkpoint {checkpoint_path}: {e}")

        print(f"Removed old checkpoints after {self.trial_counter} trials.")




"""
Ray Trial-Name Creator
-------------------------------------------------------------------------------------------------------------------------------------------
Creates a shorter name for the trials and their directories to avoid an exception on windows where the path is too long,
due to too many search parameters (name usually is concatenation of key:value and date).
"""
def custom_trial_dir_name(trial: Trial):
    return f'trial_{trial.trial_id}'