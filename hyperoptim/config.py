
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import ray.tune.search.sample as sample

@dataclass
class DatasetConfig:

    dataset_kind: str  = 'key'
    normaliser_kind: str = 'raw'
    exclude_columns: list[str] = field(default_factory = lambda: [])
    train_size: float = 0.9

    def __str__(self):
        return (
            "DatasetConfig(\n"
            f"  dataset_kind='{self.dataset_kind}',\n"
            f"  normaliser_kind='{self.normaliser_kind}',\n"
            f"  exclude_columns={self.exclude_columns},\n"
            f"  train_size={self.train_size}\n"
            ")"
        )




@dataclass
class ExperimentConfig:

    experiment_name: str
    optim_loss: str
    optim_mode: str
    num_samples: int
    search_space: dict
    trainable: Callable
    #loss_name: Optional[str] = None
    eval_metrics: list[str] = field(default_factory = lambda: [])
    data_cfg: DatasetConfig = field(default_factory = DatasetConfig)
    model_params: dict[str, Any] = field(default_factory = lambda: {})

    def __str__(self):
        search_space_str = self.format_search_space()
        return (
            "ExperimentConfig(\n"
            f"  experiment_name='{self.experiment_name}',\n"
            f"  optim_loss='{self.optim_loss}',\n"
            f"  optim_mode='{self.optim_mode}',\n"
            f"  num_samples={self.num_samples},\n"
            f"  search_space={{\n{search_space_str}\n  }},\n"
            f"  trainable={self.trainable.__name__},\n"
            f"  eval_metrics={self.eval_metrics},\n"
            f"  data_cfg={self.data_cfg},\n"
            f"  model_params={self.model_params}\n"
            ")"
        )


    def format_search_space(self) -> str:
        lines = []
        for key, value in self.search_space.items():
            if isinstance(value, sample.Categorical):
                choices = value.categories
                line = f"    '{key}': tune.choice({choices}),"
            elif isinstance(value, sample.Integer):
                line = f"    '{key}': tune.randint(lower={value.lower}, upper={value.upper}),"
            elif isinstance(value, sample.Float):
                if isinstance(value, sample.LogUniform):
                    line = f"    '{key}': tune.loguniform({value.lower}, {value.upper}),"
                else:
                    line = f"    '{key}': tune.uniform({value.lower}, {value.upper}),"
            else:
                line = f"    '{key}': {value},"
            lines.append(line)
        
        return '\n'.join(lines)