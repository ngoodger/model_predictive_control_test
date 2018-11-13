from random import random, seed

import block_sys as bs
import numpy as np
from block_sys import FORCE_SCALE, FRAMES, GRID_SIZE, IMAGE_DEPTH
from torch.utils.data import Dataset

# MEAN_S0 = (BLOCK_SIZE ** 2) / (GRID_SIZE ** 2)
MEAN_S0 = 0.


class ModelDataSet(Dataset):
    def __init__(self, size, seq_len, random_seed=0):
        super(ModelDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.size = size
        self.seq_len = seq_len
        seed(random_seed)

    def __len__(self):
        return self.size

    def _random_force(self):
        force = np.array([random() - 0.5, random() - 0.5], dtype=np.float32)
        return force

    def __getitem__(self, idx):
        self.my_block_sys.reset()
        forces = []
        observations = []
        # Generate seq_len + 1 quad frames since minimum 2 (input + label).
        for seq_idx in range(self.seq_len + 1):
            observation_item = np.zeros(
                [IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32
            )
            force_item = np.zeros([2], dtype=np.float32)
            force_item[:] = self._random_force()
            # Collect 4 frames
            for i in range(FRAMES):
                observation_item[0, :, :, i] = (
                    self.my_block_sys.step(
                        FORCE_SCALE * force_item[0], FORCE_SCALE * force_item[1]
                    )
                    - MEAN_S0
                )
            forces.append(force_item)
            observations.append(observation_item)

        return (forces, observations)


class PolicyDataSet(Dataset):
    def __init__(self, size):
        super(PolicyDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        # Only one frame is used as target.
        self.size = size

    def __len__(self):
        return self.size

    def _random_force(self):
        force = np.array([random() - 0.5, random() - 0.5], dtype=np.float32)
        return force

    def __getitem__(self, idx):
        seed(idx)
        start = np.zeros([IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32)
        target = np.zeros([IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32)
        force_0 = np.zeros([2], dtype=np.float32)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        force_0[:] = self._random_force()
        for i in range(FRAMES):
            start[0, :, :, i] = (
                self.my_block_sys.step(
                    FORCE_SCALE * force_0[0], FORCE_SCALE * force_0[1]
                )
                - MEAN_S0
            )
        # Get a single target frame.
        # A little bit wasteful in we are running uneccessary steps.
        # But data generation parallelized we don't care until our cpu is
        # fully utilized.
        self.my_block_sys.reset()
        frame = self.my_block_sys.step(0., 0.)
        target[0, :, :, 0] = frame
        target[0, :, :, 1] = frame
        target[0, :, :, 2] = frame
        target[0, :, :, 3] = frame

        return (force_0, start, target)
