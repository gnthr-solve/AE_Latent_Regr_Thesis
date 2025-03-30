
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from typing import Optional, Tuple


def get_quantile_limits(tensor: Tensor, lower=5, upper=95, buffer=0.2):
    q_low = torch.quantile(tensor, lower/100)
    q_high = torch.quantile(tensor, upper/100)
    span = q_high - q_low
    return (q_low - buffer*span, q_high + buffer*span)

"""
LatentSpaceVisualiser
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LatentSpaceVisualiser:

    frame_extension = 'jpg'

    def __init__(
        self,
        output_dir: str = "./latent_frames",
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 3,
        dpi: int = 300,
        title: str = 'Latent Space Evolution',
        iter_label: str = 'Epoch',
        cmap: str = 'viridis',
        lims_by: str = 'quantile',
        lims_span_buffer: float = 0.1,
    ):
        
        self.output_dir = output_dir
        self.frame_size = (frame_width, frame_height)
        self.fps = fps
        self.dpi = dpi

        self.title = title
        self.iter_label = iter_label
        self.cmap = plt.get_cmap(cmap)
        self.lims_by = lims_by
        self.lims_span_buffer = lims_span_buffer
        
        self.latent_history: dict[int, Tensor] = {}
        self.loss_history: dict[int, float] = {}

        # Set up directory structure
        os.makedirs(self.output_dir, exist_ok=True)


    def __call__(
        self,
        latent_tensors: Tensor,
        iteration: Optional[int] = None,
        loss: Optional[Tensor] = None,
    ):
        
        self.latent_history[iteration] = latent_tensors.detach().clone()
        self.loss_history[iteration] = loss.detach().item() if loss is not None else None


    def create_frames(self):

        # Initialize plot style
        plt.style.use('bmh')
        frame_width, frame_height = self.frame_size
        self.fig = plt.figure(figsize=(frame_width/100, frame_height/100), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        self.determine_lims()

        latent_history = {k: lt.numpy() for k, lt in self.latent_history.items()}
        total_iters = len(latent_history)

        for iteration, latent_vectors in latent_history.items():

            norm_iter = iteration / total_iters
            color = self.cmap(norm_iter)

            loss = self.loss_history.get(iteration, None)

            self.ax.clear()
            
            self.ax.scatter(
                latent_vectors[:, 0],
                latent_vectors[:, 1],
                c=color,
                s = 40,
                alpha=0.7,
                edgecolors='w',
                linewidths=0.2
            )

            # Add annotations
            title = self.title
            if iteration is not None:
                title += f" | {self.iter_label}: {iteration}"
            if loss is not None:
                title += f" | Loss: {loss:.4f}"
            self.ax.set_title(title, fontdict={'fontsize': 8})
            
            self.ax.set_xlabel("$z_1$")
            self.ax.set_ylabel("$z_2$")
            self.ax.grid(True, alpha=0.3)

            self.ax.set_xlim(self.xlim[0], self.xlim[1])
            self.ax.set_ylim(self.ylim[0], self.ylim[1])

            # Save frame
            frame_path = os.path.join(self.output_dir, f"frame_{iteration:04d}.{self.frame_extension}")
            self.fig.savefig(frame_path, bbox_inches='tight')
        
        plt.close(self.fig)


    def determine_lims(self):

        all_latent = torch.cat([lt for lt in self.latent_history.values()], dim = 0)
        
        if self.lims_by == 'quantile':
            # X-axis limits
            self.xlim = get_quantile_limits(all_latent[:, 0], buffer = self.lims_span_buffer)
            
            # Y-axis limits
            self.ylim = get_quantile_limits(all_latent[:, 1], buffer = self.lims_span_buffer)
            
            # Convert to numpy floats for matplotlib
            self.xlim = (self.xlim[0].item(), self.xlim[1].item())
            self.ylim = (self.ylim[0].item(), self.ylim[1].item())

        else:
            xlim = (all_latent[:, 0].min(), all_latent[:, 0].max())
            ylim = (all_latent[:, 1].min(), all_latent[:, 1].max())
            
            buffer_x = 0.1 * (xlim[1] - xlim[0])
            buffer_y = 0.1 * (ylim[1] - ylim[0])
            
            self.xlim = (xlim[0]-buffer_x, xlim[1]+buffer_x)
            self.ylim = (ylim[0]-buffer_y, ylim[1]+buffer_y)
        


    def finalize(self, output_path: str = "latent_space_evolution.mp4"):
        """Compile frames into video and clean up"""

        self.create_frames()
        iterations = list(self.latent_history.keys())

        video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            self.frame_size
        )

        # Read frames in order
        for frame_num in iterations:

            frame_path = os.path.join(self.output_dir, f"frame_{frame_num:04d}.{self.frame_extension}")

            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, self.frame_size)

            video.write(frame)

        video.release()
        
        # Cleanup
        for frame_num in iterations:
            os.remove(os.path.join(self.output_dir, f"frame_{frame_num:04d}.{self.frame_extension}"))

        os.rmdir(self.output_dir)
        
        cv2.destroyAllWindows()

        print(f"Video saved to {output_path}")




"""
LatentSpaceVisualiserLive
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LatentSpaceVisualiserLive:
    def __init__(
        self,
        output_dir: str = "./latent_frames",
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: int = 3,
        dpi: int = 300,
        fixed_limits: Optional[Tuple[float, float, float, float]] = None,
        title: str = 'Latent Space Evolution',
        iter_label: str = 'Epoch',
    ):
        
        self.output_dir = output_dir
        self.frame_size = (frame_width, frame_height)
        self.fps = fps
        self.dpi = dpi

        self.title = title
        self.iter_label = iter_label

        self.frame_count = 0
        
        # Set up directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize plot style
        plt.style.use('bmh')
        self.fig = plt.figure(figsize=(frame_width/dpi, frame_height/dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        
        # Configure axis limits
        self.fixed_limits = fixed_limits
        self.xlim = (np.inf, -np.inf)
        self.ylim = (np.inf, -np.inf)


    def __call__(
        self,
        latent_vectors: Tensor,
        iteration: Optional[int] = None,
        loss: Optional[Tensor] = None,
    ):
        """Update visualization with new latent space data"""
        latent_vectors = latent_vectors.detach().numpy()
        loss = loss.detach().item()

        self.ax.clear()
        
        self.ax.scatter(
            latent_vectors[:, 0],
            latent_vectors[:, 1],
            c='blue',
            alpha=0.7,
            edgecolors='w',
            linewidths=0.5
        )

        # Update axis limits
        if self.fixed_limits is None:
            curr_xlim = (latent_vectors[:, 0].min(), latent_vectors[:, 0].max())
            curr_ylim = (latent_vectors[:, 1].min(), latent_vectors[:, 1].max())
            
            self.xlim = (min(self.xlim[0], curr_xlim[0]), 
                        max(self.xlim[1], curr_xlim[1]))
            self.ylim = (min(self.ylim[0], curr_ylim[0]), 
                        max(self.ylim[1], curr_ylim[1]))
            
            buffer_x = 0.1 * (self.xlim[1] - self.xlim[0])
            buffer_y = 0.1 * (self.ylim[1] - self.ylim[0])
            
            self.ax.set_xlim(self.xlim[0]-buffer_x, self.xlim[1]+buffer_x)
            self.ax.set_ylim(self.ylim[0]-buffer_y, self.ylim[1]+buffer_y)

        else:
            self.ax.set_xlim(*self.fixed_limits[:2])
            self.ax.set_ylim(*self.fixed_limits[2:])

        # Add annotations
        title = self.title
        if iteration is not None:
            title += f" | {self.iter_label}: {iteration}"
        if loss is not None:
            title += f" | Loss: {loss:.4f}"
        self.ax.set_title(title)
        
        self.ax.set_xlabel("$z_1$")
        self.ax.set_ylabel("$z_2$")
        self.ax.grid(True, alpha=0.3)

        # Save frame
        frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
        self.fig.savefig(frame_path, bbox_inches='tight')
        self.frame_count += 1


    def finalize(self, output_path: str = "latent_space_evolution.mp4"):
        """Compile frames into video and clean up"""

        video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            self.frame_size
        )

        # Read frames in order
        for frame_num in range(self.frame_count):

            frame_path = os.path.join(self.output_dir, f"frame_{frame_num:04d}.png")

            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, self.frame_size)

            video.write(frame)

        video.release()
        
        # Cleanup
        for frame_num in range(self.frame_count):
            os.remove(os.path.join(self.output_dir, f"frame_{frame_num:04d}.png"))
        os.rmdir(self.output_dir)
        
        plt.close(self.fig)
        cv2.destroyAllWindows()

        print(f"Video saved to {output_path}")



