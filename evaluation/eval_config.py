
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    """
    Config for EvaluationVisitor's specifying the keys of the data and output to use 
    and the mode of model application.
    """
    data_key: str
    output_name: str
    mode: str
    description: str = ''