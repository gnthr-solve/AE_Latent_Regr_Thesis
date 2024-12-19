
import os
import shutil
import pandas as pd
import glob

import ray.tune as tune

from ray.tune import Callback
from ray.tune.experiment import Trial

from pathlib import Path




class PeriodicSaveCallback(Callback):

    def __init__(self, save_frequency, experiment_name, tracked_metrics: list[str]):
        self.experiment_name = experiment_name
        self.tracked_metrics = tracked_metrics
        
        self.save_frequency = save_frequency
        self.trial_counter = 0
        
        self.results_dir = Path(f'./results/{experiment_name}/')
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




class GlobalBestModelSaver(Callback):

    def __init__(self, tracked_metric, mode, cleanup_frequency, experiment_name):
        self.experiment_name = experiment_name
        
        self.tracked_metric = tracked_metric
        self.mode = mode 
        self.cleanup_frequency = cleanup_frequency
        self.trial_counter = 0
        
        self.global_best_metric = None
        self.global_best_checkpoint_path = None
        
        self.results_dir = f'./results/{experiment_name}/'
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

                # Delete the previous global best model checkpoint if it exists
                if self.global_best_checkpoint_path and os.path.exists(self.global_best_checkpoint_path):
                    os.remove(self.global_best_checkpoint_path)

                # Save the new global best model checkpoint
                checkpoint = trial.checkpoint
                if checkpoint:
                    with checkpoint.as_directory() as checkpoint_dir:
                        # Copy the model file to the results directory
                        model_file_name = 'model.pt'

                        src_model_path = os.path.join(checkpoint_dir, model_file_name)
                        dest_model_path = os.path.join(self.results_dir, f'best_model.pt')

                        shutil.copyfile(src_model_path, dest_model_path)

                        self.global_best_checkpoint_path = dest_model_path

                        print(f"New global best model with {self.tracked_metric}: {self.global_best_metric} saved to {dest_model_path}")
        
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



def custom_trial_dir_name(trial):
    return f'trial_{trial.trial_id}'