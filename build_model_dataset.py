import block_sys as bs
import numpy as np
from random import random
# import curses
import pickle
import torch.utils.data

try:
    SAMPLE_COUNT = 10000
    IMAGE_DEPTH = 1
    SAMPLE_PRINT_INTERVAL = 1000
    my_block_sys = bs.BlockSys(use_curses=True)
    # State is 4 Grids to satisfy markov property
    s0 = np.zeros([SAMPLE_COUNT, IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                  dtype=np.float32)
    s1 = np.zeros([SAMPLE_COUNT, IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                  dtype=np.float32)
    force = np.zeros([SAMPLE_COUNT, 2], dtype=np.float32)
    samples = []
    for sample_idx in range(SAMPLE_COUNT):
        if sample_idx % SAMPLE_PRINT_INTERVAL == 0:
            print(sample_idx)
        my_block_sys.reset()

        # Collect 4 initial frames (s0)
        for i in range(4):
            observation = my_block_sys.step(bs.FORCE_SCALE * (random() - 0.5),
                                            bs.FORCE_SCALE * (random() - 0.5))
            s0[sample_idx, :, :,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation
            force[sample_idx, :] = (np.array([bs.FORCE_SCALE *
                                    (random() - 0.5),
                                    bs.FORCE_SCALE * (random() - 0.5)],
                                    dtype=np.float32))

        # Collect 4 following frames (s1)
        for i in range(4):
            observation = my_block_sys.step(force[sample_idx, 0],
                                            force[sample_idx, 1])
            s1[sample_idx, :, :,
               (i * bs.GRID_SIZE):
               ((i * bs.GRID_SIZE) + bs.GRID_SIZE)] = observation
        """
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

        # samples.append({"s0": s0, "force": force, "s1": s1})
    s0_tensor = torch.from_numpy(s0)
    force_tensor = torch.from_numpy(force)
    s1_tensor = torch.from_numpy(s1)
    samples = torch.utils.data.TensorDataset(s0_tensor,
                                             force_tensor, s1_tensor)
    pickle.dump(s0, open("s0.p", "wb"))
    pickle.dump(force, open("force.p", "wb"))
    pickle.dump(s1, open("s1.p", "wb"))
finally:
    pass
#    curses.echo()
#    curses.nocbreak()
#    curses.endwin()
