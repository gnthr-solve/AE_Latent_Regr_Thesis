
from dataclasses import dataclass, field

@dataclass
class EvalConfig:

    data_key: str
    output_name: str
    mode: str
    loss_name: str
    description: str = ''