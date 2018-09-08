import block_sys as bs
from torch.utils.data import Dataset
from block_sys import IMAGE_DEPTH, FORCE_SCALE, GRID_SIZE, BLOCK_SIZE
from random import random
from random import seed
import numpy as np

MEAN_S0 = (BLOCK_SIZE ** 2) / (GRID_SIZE ** 2)

class BlockDataSet(Dataset):
    def __init__(self, size):
        super(BlockDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, bs.GRID_SIZE],
                           dtype=np.float32)
        self.s1 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, bs.GRID_SIZE],
                           dtype=np.float32)
        self.force_0 = np.zeros([2], dtype=np.float32)
        self.force_1 = np.zeros([2], dtype=np.float32)
        self.size = size

    def __len__(self):
        return self.size

    def _random_force(self):
        force = (np.array([random() - 0.5, random() - 0.5],
                          dtype=np.float32))
        return force

    def __getitem__(self, idx):
        seed(idx)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        self.force_0[:] = self._random_force()
        for i in range(4):
            self.s0[i, :, :] = (self
                                .my_block_sys
                                .step(FORCE_SCALE * self.force_0[0],
                                      FORCE_SCALE * self.force_0[1])
                                - MEAN_S0)

        self.force_1[:] = self._random_force()
        # Collect 4 following frames (s1)
        for i in range(4):
            self.s1[i, :, :] = (self
                                .my_block_sys
                                .step(FORCE_SCALE * self.force_1[0],
                                      FORCE_SCALE * self.force_1[1]))

        return (self.force_0, self.s0, self.force_1, self.s1)


class BlockDataSetSimple(Dataset):
    def __init__(self, size):
        super(BlockDataSetSimple, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros(8,
                           dtype=np.float32)
        self.s1 = np.zeros(8,
                           dtype=np.float32)
        self.force_0 = np.zeros([2], dtype=np.float32)
        self.force_1 = np.zeros([2], dtype=np.float32)
        self.size = size

    def __len__(self):
        return self.size

    def _random_force(self):
        force = (np.array([random() - 0.5, random() - 0.5],
                          dtype=np.float32))
        return force

    def __getitem__(self, idx):
        seed(idx)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        self.force_0[:] = self._random_force()
        for i in range(4):
            self.s0[2 * i:2 * i + 2] = (self
                                        .my_block_sys
                                        .step_simple(FORCE_SCALE *
                                                     self.force_0[0],
                                                     FORCE_SCALE *
                                                     self.force_0[1]) /
                                        GRID_SIZE)

        self.force_1[:] = self._random_force()
        # Collect 4 following frames (s1)
        for i in range(4):
            self.s1[2 * i:2 * i + 2] = (self
                                .my_block_sys
                                .step_simple(FORCE_SCALE * self.force_1[0],
                                      FORCE_SCALE * self.force_1[1]))

        return (self.force_0, self.s0, self.force_1, self.s1)
