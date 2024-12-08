
import torch
import pandas as pd
from torch import Tensor

import matplotlib.pyplot as plt
import seaborn as sns

from .eval_visitor_abc import EvaluationVisitor
from ..eval_config import EvalConfig
from ..evaluation import Evaluation

"""
Plotting Visitors - LatentPlotVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Plots the latent Z in a 3D plot, if latent_dim = 3. Either with or without an associated error. 
"""
class LatentPlotVisitor(EvaluationVisitor):

    def __init__(self, eval_cfg: EvalConfig, w_loss: bool = True):
        super().__init__(eval_cfg = eval_cfg)

        self.w_loss = w_loss


    def visit(self, eval: Evaluation):

        model_output = eval.model_outputs[self.output_name]
        latent_tensor = model_output.Z_batch

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if self.w_loss:
            loss_tensor = eval.results.losses[self.loss_name]

            scatter = ax.scatter(latent_tensor[:, 0], latent_tensor[:, 1], latent_tensor[:, 2], c = loss_tensor, cmap = 'RdYlGn_r')

            colorbar = fig.colorbar(scatter)
            colorbar.set_label(self.loss_name)

        else:
            scatter = ax.scatter(latent_tensor[:, 0], latent_tensor[:, 1], latent_tensor[:, 2])

        
        ax.set_xlabel('$x_l$')
        ax.set_ylabel('$y_l$')
        ax.set_zlabel('$z_l$')

        title = f'Latent Space with {self.loss_name} Error' if self.w_loss else 'Latent Space'
        plt.title(title)

        plt.show()




class LatentDistributionVisitor(EvaluationVisitor):

    def visit(self, eval: Evaluation):

        Z_batch = eval.model_outputs[self.output_name].Z_batch
        latent_dim = Z_batch.shape[1]
        
        # Statistical analysis
        stats = {
            'mean': Z_batch.mean(dim=0),
            'std': Z_batch.std(dim=0),
            'range': (Z_batch.min(dim=0)[0], Z_batch.max(dim=0)[0])
        }
        
        # Calculate grid layout
        plots_per_row = 3
        n_rows = (latent_dim + plots_per_row - 1) // plots_per_row
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows = n_rows, 
            ncols = plots_per_row,  
            figsize = (5*plots_per_row, 5*n_rows),
            squeeze = False
        )
        
        # Create distribution plots
        for i in range(latent_dim):
            row = i // plots_per_row
            col = i % plots_per_row
            
            sns.histplot(Z_batch[:, i], ax=axes[row, col])
            axes[row, col].set_title(f'Dimension {i+1}')
        
        # Remove empty subplots
        for i in range(latent_dim, n_rows * plots_per_row):
            row = i // plots_per_row
            col = i % plots_per_row
            fig.delaxes(axes[row, col])
        
        fig.tight_layout()
        
        eval.results.metrics[f'{self.output_name}_latent_stats'] = stats
        eval.results.plots[f'{self.output_name}_latent_dist'] = fig

        plt.show()
