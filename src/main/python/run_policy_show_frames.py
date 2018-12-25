from random import random
import block_sys as bs
import numpy as np
import torch
from block_sys import FORCE_SCALE, FRAMES, GRID_SIZE, IMAGE_DEPTH

MEAN_S0 = 0.
USE_POLICY_SPECIFIC_INPUT_CNN = False


def run_policy_show_frames():
    TEST_STEPS = 10

    # Use cpu for inference.
    device = "cpu"
    my_block_sys = bs.BlockSys()
    my_block_sys_target = bs.BlockSys()
    model = torch.load("recurrent_model.pt", map_location=device)
    policy = torch.load("my_policy.pt", map_location=device)
    model_input_cnn = torch.load("input_cnn.pt", map_location=device)
    out_array = np.zeros([bs.GRID_SIZE, bs.GRID_SIZE, 3])
    if USE_POLICY_SPECIFIC_INPUT_CNN:
        input_cnn = torch.load("policy_input_cnn.pt", map_location=device)
        model_input_cnn = torch.load("policy_input_cnn.pt", map_location=device)
    else:
        input_cnn = torch.load("input_cnn.pt", map_location=device)
        model_input_cnn = torch.load("input_cnn.pt", map_location=device)
    force_0 = np.zeros([1, 2], dtype=np.float32)
    s0 = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32)
    s1_target = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, 4], dtype=np.float32)
    # Get Initial state
    for i in range(FRAMES):
        force_0[0, :] = np.array(
            [random() - 0.5, random() - 0.5], dtype=np.float32
        ).reshape([1, 2])
        for i in range(FRAMES):
            s0[0, 0, :, :, i] = my_block_sys.step(
                FORCE_SCALE * (force_0[0, 0]), FORCE_SCALE * (force_0[0, 1])
            )
    # Get a single target frame from target block sys.
    s1_target[0, 0, :, :, 0] = my_block_sys_target.step(0., 0.)
    s1_target[0, 0, :, :, 1] = s1_target[0, 0, :, :, 0]
    s1_target[0, 0, :, :, 2] = s1_target[0, 0, :, :, 0]
    s1_target[0, 0, :, :, 3] = s1_target[0, 0, :, :, 0]
    for i in range(TEST_STEPS):
        print(force_0)
        force_0_tensor = torch.from_numpy(force_0).to(device)
        start = torch.from_numpy(s0).to(device)
        target = torch.from_numpy(s1_target).to(device)
        out_target_cnn_flat = input_cnn.forward(target)
        if i == 0:
            out_start_cnn_flat_model = model_input_cnn.forward(start)
            out_start_cnn_flat = input_cnn.forward(start)
            force_1_tensor, out_target_cnn_layer = policy.forward(
                force_0_tensor,
                out_start_cnn_flat,
                out_target_cnn_flat,
                None,
                first_iteration=True,
            )
            logits, out, recurrent_state = model.forward(
                out_start_cnn_flat_model,
                None,
                force_0_tensor,
                force_1_tensor,
                first_iteration=True,
            )
        else:
            out_start_cnn_flat_model = model_input_cnn.forward(start)
            out_start_cnn_flat = input_cnn.forward(start)
            force_1_tensor, _ = policy.forward(
                force_0_tensor,
                out_start_cnn_flat,
                out_target_cnn_flat,
                out_target_cnn_layer=out_target_cnn_layer,
                first_iteration=False,
            )
            # DELETE THIS
            # force_1_tensor = force_0_tensor
            logits, out, recurrent_state = model.forward(
                out_start_cnn_flat_model,
                recurrent_state,
                force_0_tensor,
                force_1_tensor,
                first_iteration=False,
            )

        # Get new s0 based on policy force.
        force_1 = force_1_tensor.data.numpy()
        for j in range(FRAMES):
            s0[0, 0, :, :, j] = my_block_sys.step(
                FORCE_SCALE * force_1[0, 0], FORCE_SCALE * force_1[0, 1]
            )
        force_0 = force_1

        for j in range(FRAMES):
            out_array = np.zeros([bs.GRID_SIZE, bs.GRID_SIZE, 3])
            out_array[:, :, 2] = s1_target[0, 0, :, :, 0]
            out_array[:, :, 1] = start[0, :, :, :, j].data.numpy()
            out_array[:, :, 0] = out[0, :, :, :, j].data.numpy()
            bs.render(out_array, "e" + str(j + i*FRAMES))


if __name__ == "__main__":
    run_policy_show_frames()
