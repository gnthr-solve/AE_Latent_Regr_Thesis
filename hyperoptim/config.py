
from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class DatasetConfig:
    dataset_kind: str  = 'key'
    normaliser_kind: str = ''
    exclude_columns: list[str] = field(default_factory = lambda: [])
    train_size: float = 0.9


@dataclass
class ExperimentConfig:
    experiment_name: str
    optim_loss: str
    optim_mode: str
    num_samples: int
    search_space: dict
    trainable: Callable
    #loss_name: Optional[str] = None
    metrics: list[str] = field(default_factory = lambda: [])
    data_cfg: DatasetConfig = field(default_factory = DatasetConfig)
   