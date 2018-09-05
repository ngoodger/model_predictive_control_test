import block_sys as bs
from torch.utils.data import Dataset
from block_sys import IMAGE_DEPTH
import random
import numpy as np
import sys


class BlockDataSet(Dataset):
    def __init__(self, size):
        super(BlockDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                           dtype=np.float32)
        self.s1 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                           dtype=np.float32)
        self.force = np.zeros([2], dtype=np.float32)
        self.size = size

    def __len__(self):
        return self.size 

    def __getitem__(self, idx):
        random.seed(idx)
        self.my_block_sys.reset()
        # Collect 4 initial frames (s0)
        for i in range(4):
            observation = self.my_block_sys.step(
                                                 bs.FORCE_SCALE * (random.random() - 0.5),
                                           bs.FORCE_SCALE * (random.random() - 0.5))
            self.s0[:, :,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation
            self.force[:] = (np.array([bs.FORCE_SCALE *
                                    (random.random() - 0.5),
                                    bs.FORCE_SCALE * (random.random() - 0.5)],
                                    dtype=np.float32))
        # Collect 4 following frames (s1)
        for i in range(4):
            observation = self.my_block_sys.step(self.force[0],
                                            self.force[1])
            self.s1[:, :,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation

        return (self.s0, self.force, self.s1)
