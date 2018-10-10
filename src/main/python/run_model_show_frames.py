import block_dataset
import block_sys as bs
import torch
from torch.utils.data import DataLoader
from model import forward_sequence

SEQ_LEN = 10


def run_model_show_frames():
    TEST_EXAMPLES = 1

    # Use cpu for inference.
    samples_dataset = block_dataset.ModelDataSet(TEST_EXAMPLES, SEQ_LEN)
    dataloader = DataLoader(samples_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = torch.load("my_model.pt")
    for batch_idx, data in enumerate(dataloader):
        s = data[1]
        batch_data = {"force": data[0], "s": s, "seq_len": SEQ_LEN}
        _, y1_list = forward_sequence(model, batch_data)
        for seq_idx in range(SEQ_LEN - 1):
            for i in range(4):
                s0_frame = s[seq_idx + 1][0, :, :, :, i].data.numpy()
                bs.render(
                    s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]),
                    "_{}_{}".format(str(seq_idx), str(i) + "_target"),
                )
            for i in range(bs.FRAMES):
                y1_frame = y1_list[seq_idx][0, :, :, :, i].data.numpy()
                bs.render(
                    y1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]),
                    "_{}_{}".format(str(seq_idx), str(i) + "_out"),
                )


if __name__ == "__main__":
    run_model_show_frames()
