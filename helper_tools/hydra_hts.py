

from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from pathlib import Path
import pandas as pd
import os

class SaveResultsCallback(Callback):

    def __init__(self):
        print("SaveResultsCallback initialized")
        self.results = []


    def on_job_end(self, config, job_return, **kwargs):

        print(f"Job {config.hydra.job.name} ended with status {job_return.status}")
        
        if job_return.status == JobStatus.COMPLETED:
            
            self.results.append(job_return.return_value)

            df = pd.DataFrame(self.results)

            hydra_cfg = HydraConfig.get()
            main_dir = Path(hydra_cfg.sweep.dir)
            output_dir = main_dir / hydra_cfg.sweep.subdir

            df.to_csv(os.path.join(output_dir, f"{hydra_cfg.job.name}.csv"), index=False)


    # def on_multirun_start(self, config, **kwargs):
    #     self.results = []

    
    # def on_multirun_end(self, config, **kwargs):
    #     df = pd.DataFrame(self.results)
    #     output_dir = HydraConfig.get().run.dir
    #     os.makedirs(output_dir, exist_ok=True)
    #     df.to_csv(os.path.join(output_dir, f"{config.experiment_name}.csv"), index=False)
