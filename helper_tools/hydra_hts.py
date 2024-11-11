

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from pathlib import Path
import pandas as pd
import os

class SaveResultsCallback(Callback):

    def __init__(self):
        #print("SaveResultsCallback initialized")
        self.results = []


    def on_job_end(self, config, job_return, **kwargs):

        #print(f"Job {config.hydra.job.name} ended with status {job_return.status}")
        print(f"Config: \n{config}\n")
        if job_return.status == JobStatus.COMPLETED:
            
            self.results.append(job_return.return_value)

            df = pd.DataFrame(self.results)

            hydra_cfg = HydraConfig.get()
            main_dir = Path(hydra_cfg.sweep.dir)
            output_dir = main_dir / hydra_cfg.sweep.subdir

            df.to_csv(os.path.join(output_dir, f"{hydra_cfg.job.name}.csv"), index=False)




class SaveResultsRefactor(Callback):

    def __init__(self):
        #print("SaveResultsCallback initialized")
        self.results = []


    def on_job_end(self, config, job_return, **kwargs):

        # I don't need to pass the parameters as a return value, I can just get them from the config object
        result_dict = {param: config.get(param) for param in config.track_params}
        print(f"Result dict: \n{result_dict}\n")

        if job_return.status == JobStatus.COMPLETED:
            
            self.results.append(job_return.return_value)

            df = pd.DataFrame(self.results)

            hydra_cfg = HydraConfig.get()
            main_dir = Path(hydra_cfg.sweep.dir)
            output_dir = main_dir / hydra_cfg.sweep.subdir

            df.to_csv(os.path.join(output_dir, f"{hydra_cfg.job.name}.csv"), index=False)



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