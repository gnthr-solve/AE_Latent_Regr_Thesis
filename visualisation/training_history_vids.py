
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
class LatentSpace2DVisualiser:

    frame_extension = 'jpg'

    def __init__(
        self,
        latent_history: dict[int, Tensor],
        output_dir: str = "./results",
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
        self.frames_dir = os.path.join(output_dir, 'frames')
        self.frame_size = (frame_width, frame_height)
        self.fps = fps
        self.dpi = dpi

        self.title = title
        self.iter_label = iter_label
        self.cmap = plt.get_cmap(cmap)
        self.lims_by = lims_by
        self.lims_span_buffer = lims_span_buffer
        
        self.latent_history = {
            e*len(batches)+i: batch
            for e, batches in latent_history.items()
            for i, batch in enumerate(batches)
        }
        

        # Set up directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)


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

            #loss = self.loss_history.get(iteration, None)
            loss = None

            self.ax.clear()
            
            self.ax.scatter(
                latent_vectors[:, 0],
                latent_vectors[:, 1],
                color=color,
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
            frame_path = os.path.join(self.frames_dir, f"frame_{iteration:04d}.{self.frame_extension}")
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
        


    def finalize(self, output_name: str = "latent_space_evolution.mp4"):
        """Compile frames into video and clean up"""

        self.create_frames()
        iterations = list(self.latent_history.keys())

        output_path = os.path.join(self.output_dir, output_name)
        video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MJPG'),
            self.fps,
            self.frame_size
        )

        # Read frames in order
        for frame_num in iterations:

            frame_path = os.path.join(self.frames_dir, f"frame_{frame_num:04d}.{self.frame_extension}")

            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, self.frame_size)

            video.write(frame)

        video.release()
        
        # Cleanup
        for frame_num in iterations:
            os.remove(os.path.join(self.frames_dir, f"frame_{frame_num:04d}.{self.frame_extension}"))

        os.rmdir(self.frames_dir)
        
        cv2.destroyAllWindows()

        print(f"Video saved to {output_path}")


