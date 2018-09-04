import block_sys as bs
import numpy as np
from random import random
import pickle
import torch.utils.data
import time
import os

SAMPLE_FOLDER = "samples/"

SAMPLE_COUNT = 100000
IMAGE_DEPTH = 1
SAMPLE_PRINT_INTERVAL = 1000
my_block_sys = bs.BlockSys()
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
        with open(os.path.join(SAMPLE_FOLDER, "f_{}.txt".format(sample_idx)), "w") as f:
            f.write(str(force[sample_idx, 0]))
            f.write(str(force[sample_idx, 1]))
           
    for i in range(4):
        time.sleep(0.1)
        s0_frame = s0[sample_idx, :, :, (i * bs.GRID_SIZE):
                      (i * bs.GRID_SIZE + bs.GRID_SIZE)]
        #img = bs.render(s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
    pickle.dump(s0[sample_idx, :, :, :], open(os.path.join(SAMPLE_FOLDER, "s0_{}_{}.p".format(sample_idx, i)), "wb"))
        #img.save(os.path.join(SAMPLE_FOLDER, "s0_{}_{}.jpeg".format(sample_idx, i)))

    for i in range(4):
        time.sleep(0.1)
        s1_frame = s1[sample_idx, :, :, (i * bs.GRID_SIZE):
                      (i * bs.GRID_SIZE + bs.GRID_SIZE)]
        #img = bs.render(s1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
    pickle.dump(s1[sample_idx, :,  :, :], open(os.path.join(SAMPLE_FOLDER, "s1_{}_{}.p".format(sample_idx, i)), "wb"))
        #img.save(os.path.join(SAMPLE_FOLDER, "s1_{}_{}.jpeg".format(sample_idx, i)))

    # samples.append({"s0": s0, "force": force, "s1": s1})
s0_tensor = torch.from_numpy(s0)
force_tensor = torch.from_numpy(force)
s1_tensor = torch.from_numpy(s1)
samples = torch.utils.data.TensorDataset(s0_tensor,
                                         force_tensor, s1_tensor)
pickle.dump(s0, open("s0.p", "wb"))
pickle.dump(force, open("force.p", "wb"))
pickle.dump(s1, open("s1.p", "wb"))
