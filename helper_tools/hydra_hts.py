
import torch
import json

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from pathlib import Path
import pandas as pd
import os



class SaveResultsTable(Callback):

    def __init__(self, tracked_results: list[str]):
        
        self.tracked_results = tracked_results
        self.results = []
        

    def on_job_end(self, config, job_return, **kwargs):

        result_params = {param: config.get(param) for param in config.hyper_params}

        if job_return.status == JobStatus.COMPLETED:
            
            tracked_result = {
                name: value 
                for name, value in job_return.return_value.items() 
                if name in self.tracked_results
            }

            self.results.append({**result_params, **tracked_result})

            df = pd.DataFrame(self.results)

            output_dir = self.get_output_dir()

            df.to_csv(os.path.join(output_dir, f"{config.hydra.job.name}.csv"), index=False)


    def get_output_dir(self):

        hydra_cfg = HydraConfig.get()
        main_dir = Path(hydra_cfg.sweep.dir)
        output_dir = main_dir / hydra_cfg.sweep.subdir

        return output_dir




class BestModelCallback(Callback):

    def __init__(self, tracked_loss: str, tracked_model: str):

        self.tracked_loss = tracked_loss
        self.tracked_model = tracked_model

        self.best_loss = float('inf')
        self.best_model = None
        self.best_model_params = None


    def on_job_end(self, config, job_return, **kwargs):

        result_params = {param: config.get(param) for param in config.hyper_params}

        if job_return.status == JobStatus.COMPLETED:
            
            loss = job_return.return_value[self.tracked_loss]

            if loss < self.best_loss:

                self.best_loss = loss
                self.best_model = job_return.return_value[self.tracked_model]
                self.best_model_params = result_params

                result_dict = {**result_params, self.tracked_loss: loss}

                ###- Save model and params -###
                output_dir = self.get_output_dir()
                name = f'best_{self.tracked_model}'

                torch.save(self.best_model.state_dict(), os.path.join(output_dir, f"{name}.pth"))

                with open(os.path.join(output_dir, f"{name}_params.json"), 'w') as f:
                    json.dump(result_dict, f)


    def get_output_dir(self):

        hydra_cfg = HydraConfig.get()
        main_dir = Path(hydra_cfg.sweep.dir)
        output_dir = main_dir / hydra_cfg.sweep.subdir

        return output_dir




"""
I could create callback hook classes for exporting observer data, for exporting the model based on conditions, or similar.
Hooks could have an interface like 
class ExportHook:

    def __call__(self, config, job_return):
        
        output_dir = self.get_export_path(config)
        export_product = self.obtain_result(job_return.return_value)

        self.export(export_product, output_dir)

        
    @abstract_method
    def obtain_result(self, return_dict):
        pass
    
    @abstract_method
    def export(self, export_product, output_dir):
        pass
        
    def get_export_path(self, config):
    
        hydra_cfg = HydraConfig.get()
        main_dir = Path(hydra_cfg.sweep.dir)
        output_dir = main_dir / hydra_cfg.sweep.subdir

        return output_dir


where each hook would know what to get from the return value and how to export it.
"""