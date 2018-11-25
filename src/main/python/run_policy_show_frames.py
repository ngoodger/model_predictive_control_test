from random import random

import block_sys as bs
import numpy as np
import torch
from block_sys import FORCE_SCALE, FRAMES, GRID_SIZE, IMAGE_DEPTH

MEAN_S0 = 0.


def run_policy_show_frames():
    TEST_STEPS = 100

    # Use cpu for inference.
    device = "cpu"
    my_block_sys = bs.BlockSys()
    my_block_sys_target = bs.BlockSys()
    policy = torch.load("my_policy.pt", map_location="cpu")
    input_cnn = torch.load("input_cnn.pt", map_location="cpu")
    force_0 = np.zeros([1, 2], dtype=np.float32)
    s0 = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32)
    s1_target = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, 2], dtype=np.float32)
    # Get Initial state
    for i in range(FRAMES):
        force_0[0, :] = np.array(
            [random() - 0.5, random() - 0.5], dtype=np.float32
        ).reshape([1, 2])
        for i in range(FRAMES):
            s0[0, 0, :, :, i] = (
                my_block_sys.step(
                    FORCE_SCALE * force_0[0, 0], FORCE_SCALE * force_0[0, 1]
                )
                - MEAN_S0
            )
    # Get a single target frame from target block sys.
    s1_target[0, 0, :, :, 0] = my_block_sys_target.step(0., 0.)
    s1_target[0, 0, :, :, 1] = s1_target[0, 0, :, :, 1]
    # Save frames to disk.
    # Only 2 identical frames used as label so we can just save 1.
    for i in range(1):
        s1_frame = s1_target[0, :, :, :, i]
        bs.render(s1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
    for i in range(TEST_STEPS):
        force_0_tensor = torch.from_numpy(force_0).to(device)
        start = torch.from_numpy(s0).to(device)
        target = torch.from_numpy(s1_target).to(device)
        out_target_cnn_flat = input_cnn.forward(target)
        out_start_cnn_flat = input_cnn.forward(start)
        # batch_data = {"start": start, "target": target, "force_0": force_0_tensor}
        if i == 0:
            force_1, out_target_cnn_layer = policy.forward(
                force_0_tensor,
                out_start_cnn_flat,
                out_target_cnn_flat,
                None,
                first_iteration=True,
            )
        else:
            force_1, _ = policy.forward(
                force_0_tensor,
                out_start_cnn_flat,
                out_target_cnn_flat,
                out_target_cnn_layer=out_target_cnn_layer,
                first_iteration=False,
            )

        for i in range(FRAMES):
            s0_frame = start[0, :, :, :, i].data.numpy()
            # print(s0_batch.shape)
            bs.render(s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))

        # Get new s0 based on policy force.
        force_0 = force_1.data.numpy()
        for i in range(FRAMES):
            s0[0, 0, :, :, i] = (
                my_block_sys.step(
                    FORCE_SCALE * force_0[0, 0], FORCE_SCALE * force_0[0, 1]
                )
                - MEAN_S0
            )


if __name__ == "__main__":
    run_policy_show_frames()
