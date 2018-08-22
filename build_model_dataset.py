import block_sys as bs
import numpy as np
from random import random
import time
import curses
import pickle

try:
    SAMPLE_COUNT = 1000000
    SAMPLE_PRINT_INTERVAL = 1000
    my_block_sys = bs.BlockSys(use_curses=True)
    # State is 4 Grids to satisfy markov property
    s0 = np.zeros([bs.GRID_SIZE, 4 * bs.GRID_SIZE], dtype=np.float32)
    s1 = np.zeros([bs.GRID_SIZE, 4 * bs.GRID_SIZE], dtype=np.float32)
    force = np.zeros([2], dtype=np.float32)
    samples = []
    for sample_idx in range(SAMPLE_COUNT):
        if sample_idx % SAMPLE_PRINT_INTERVAL == 0:
            print(sample_idx)
        my_block_sys.reset()

        # Collect 4 initial frames (s0)
        for i in range(4):
            observation = my_block_sys.step(bs.FORCE_SCALE * (random() - 0.5),
                                            bs.FORCE_SCALE * (random() - 0.5))
            s0[:,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation
            force[:] = (np.array([bs.FORCE_SCALE * (random() - 0.5),
                                 bs.FORCE_SCALE * (random() - 0.5)],
                                 dtype=np.float32))

        # Collect 4 following frames (s1)
        """
        for i in range(4):
            observation = my_block_sys.step(force[0], force[1])
            s1[:,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation

        for i in range(4):
            time.sleep(0.1)
            s0_frame = s0[:, (i * bs.GRID_SIZE):
                          (i * bs.GRID_SIZE + bs.GRID_SIZE)]
            my_block_sys.show(np.rint(s0_frame).astype(np.int64))

        for i in range(4):
            time.sleep(0.1)
            s1_frame = s1[:, (i * bs.GRID_SIZE):
                          (i * bs.GRID_SIZE + bs.GRID_SIZE)]
            my_block_sys.show(2 * np.rint(s1_frame).astype(np.int64))
        """

        samples.append({"s0": s0, "force": force, "s1": s1})
    pickle.dump(samples, open("samples.p", "wb"))
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()
