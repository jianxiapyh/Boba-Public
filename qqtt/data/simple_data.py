# The simplest test data with full 3D point trajectories (n_frames, n_points, 3)
import numpy as np
import torch
from qqtt.utils import logger, visualize_pc, cfg


class SimpleData:
    def __init__(self, visualize=False):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")

        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        self.data = np.load(self.data_path)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=cfg.device)
        self.frame_len = self.data.shape[0]
        self.point_num = self.data.shape[1]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize)

    def visualize_data(self, visualize=False):
        if visualize:
            visualize_pc(
                self.data,
                visualize=True,
            )
        visualize_pc(
            self.data,
            visualize=False,
            save_video=True,
            save_path=f"{self.base_dir}/gt.mp4",
        )
