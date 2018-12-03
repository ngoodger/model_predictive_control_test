import block_dataset
import block_sys as bs
import torch
from torch.utils.data import DataLoader
from model import forward_sequence
import numpy as np

PRINT_SIM = True
PRINT_MODEL = True
SEQ_LEN = 100


def run_model_show_frames():
    TEST_EXAMPLES = 1

    # Use cpu for inference.
    SEED = 2
    samples_dataset = block_dataset.ModelDataSet(TEST_EXAMPLES, SEQ_LEN, SEED)
    dataloader = DataLoader(samples_dataset, batch_size=1, shuffle=False, num_workers=0)
    my_input_cnn = torch.load("input_cnn.pt", map_location="cpu")
    model = torch.load("recurrent_model.pt", map_location="cpu")
    out_array = np.zeros([bs.GRID_SIZE, bs.GRID_SIZE, 3])
    for batch_idx, data in enumerate(dataloader):
        forces, observations = data
        batch_data = {
            "forces": forces,
            "observations": observations,
            "seq_len": SEQ_LEN,
        }
        _, y1_list = forward_sequence(my_input_cnn, model, batch_data)
        for seq_idx in range(SEQ_LEN - 1):
            if PRINT_SIM:
                for i in range(4):
                    # s0_frame = observations[seq_idx + 1][0, :, :, :, i].data.numpy()
                    out_array[:, :, 1] = observations[seq_idx + 1][
                        0, :, :, :, i
                    ].data.numpy()
                    out_array[:, :, 0] = y1_list[seq_idx][0, :, :, :, i].data.numpy()
                    bs.render(out_array, "_{}_{}".format(str(seq_idx), str(i) + "_out"))


if __name__ == "__main__":
    run_model_show_frames()
