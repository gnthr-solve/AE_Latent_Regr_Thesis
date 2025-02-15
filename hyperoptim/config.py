
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import ray.tune.search.sample as sample

@dataclass
class DatasetConfig:
    """
    Config for TensorDataset instantiation.

    Members:
    ---------
        dataset_kind: str
            Whether to use KEY or MAX dataset.
        normaliser_kind: str
            Name of normaliser to be instantiated.
        exclude_columns: list[str]
            Columns to be removed from the TensorDataset.
        train_size: float = 0.9
            Size of training dataset subset in train-test-split.
    """
    dataset_kind: str  = 'key'
    normaliser_kind: str = 'raw'
    exclude_columns: list[str] = field(default_factory = lambda: [])
    train_size: float = 0.9

    def __str__(self):
        """
        String representation for logging.
        """
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
    """
    Ray Tune experiment config dataclass.

    Members:
    ---------
        experiment_name: str
            Name of experiment for directory labelling.
        optim_loss: str
            Name of loss term that is to be optimised.
        optim_mode: str
            Mode of optimisation, either 'max' or 'min'.
        num_samples: int
            Number of trials per experiment.
        search_space: dict
            Dictionary of initial distributions for each hyperparameter.
            Example member: 'epochs': tune.randint(...)
        trainable: Callable
            Training function to be optimised. Should report optim_loss.
        eval_metrics: list[str]
            List of names of additional LossTerm metrics to be reported during evaluation.
        data_cfg: DatasetConfig
            Dataset config for dataset instantiation.
        model_params: dict
            Dictionary to specify which model to choose in trainables where multiple are available (e.g. AE | NVAE).
    """
    experiment_name: str
    optim_loss: str
    optim_mode: str
    num_samples: int
    search_space: dict
    trainable: Callable
    eval_metrics: list[str] = field(default_factory = lambda: [])
    data_cfg: DatasetConfig = field(default_factory = DatasetConfig)
    model_params: dict[str, Any] = field(default_factory = lambda: {})

    def __str__(self):
        """
        String representation for logging.
        """
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
        """
        Formats search space for experiment logging.
        """
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