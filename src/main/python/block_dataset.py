from random import random, seed

import block_sys as bs
import numpy as np
from block_sys import FORCE_SCALE, FRAMES, GRID_SIZE, IMAGE_DEPTH
from torch.utils.data import Dataset

# MEAN_S0 = (BLOCK_SIZE ** 2) / (GRID_SIZE ** 2)
MEAN_S0 = 0.


class ModelDataSet(Dataset):
    def __init__(self, size):
        super(ModelDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros(
            [IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32
        )
        self.s1 = np.zeros(
            [IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32
        )
        self.force_0 = np.zeros([2], dtype=np.float32)
        self.force_1 = np.zeros([2], dtype=np.float32)
        self.size = size

    def __len__(self):
        return self.size

    def _random_force(self):
        force = np.array([random() - 0.5, random() - 0.5], dtype=np.float32)
        return force

    def __getitem__(self, idx):
        seed(idx)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        self.force_0[:] = self._random_force()
        for i in range(FRAMES):
            self.s0[0, :, :, i] = (
                self.my_block_sys.step(
                    FORCE_SCALE * self.force_0[0], FORCE_SCALE * self.force_0[1]
                )
                - MEAN_S0
            )

        self.force_1[:] = self._random_force()
        # Collect 4 following frames (s1)
        for i in range(FRAMES):
            self.s1[0, :, :, i] = self.my_block_sys.step(
                FORCE_SCALE * self.force_1[0], FORCE_SCALE * self.force_1[1]
            )

        return (self.force_0, self.s0, self.force_1, self.s1)


class PolicyDataSet(Dataset):
    def __init__(self, size):
        super(PolicyDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros(
            [IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32
        )
        # Only one frame is used as target.
        self.s1 = np.zeros([IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, 1], dtype=np.float32)
        self.force_0 = np.zeros([2], dtype=np.float32)
        self.size = size

    def __len__(self):
        return self.size

    def _random_force(self):
        force = np.array([random() - 0.5, random() - 0.5], dtype=np.float32)
        return force

    def __getitem__(self, idx):
        seed(idx)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        self.force_0[:] = self._random_force()
        for i in range(FRAMES):
            self.s0[0, :, :, i] = (
                self.my_block_sys.step(
                    FORCE_SCALE * self.force_0[0], FORCE_SCALE * self.force_0[1]
                )
                - MEAN_S0
            )
        # Get a single target frame.
        # A little bit wasteful in we are running uneccessary steps.
        # But data generation parallelized we don't care until our cpu is
        # fully utilized.
        self.my_block_sys.reset()
        self.s1[0, :, :, 0] = self.my_block_sys.step(0., 0.)

        return (self.force_0, self.s0, self.s1)
