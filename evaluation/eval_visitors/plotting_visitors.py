
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt

from .eval_visitor_abc import EvaluationVisitor
from ..evaluation import Evaluation


class LatentPlotVisitor(EvaluationVisitor):

    def visit(self, eval: Evaluation):

        model_output = eval.model_outputs[self.output_name]
        latent_tensor = model_output.Z_batch

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(latent_tensor[:, 0], latent_tensor[:, 1], latent_tensor[:, 2])

        # Set plot labels and title
        ax.set_xlabel('$x_l$')
        ax.set_ylabel('$y_l$')
        ax.set_zlabel('$z_l$')
        plt.title('Latent Space')

        plt.show()