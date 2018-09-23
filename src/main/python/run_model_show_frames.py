import block_dataset
import block_sys as bs
import torch
from torch.utils.data import DataLoader


def run_model_show_frames():
    TEST_EXAMPLES = 10

    # Use cpu for inference.
    device = "cpu"
    samples_dataset = block_dataset.ModelDataSet(TEST_EXAMPLES)
    dataloader = DataLoader(samples_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = torch.load("my_model.pt")
    for batch_idx, data in enumerate(dataloader):
        force_0_batch = data[0].to(device)
        s0_batch = data[1].to(device)
        force_1_batch = data[2].to(device)
        s1_batch = data[3].to(device)
        batch_data = {
            "s0": s0_batch,
            "s1": s1_batch,
            "force_0": force_0_batch,
            "force_1": force_1_batch,
        }
        _, y1 = model.forward(batch_data)
        for i in range(4):
            s0_frame = s0_batch[0, :, :, :, i].data.numpy()
            print(s0_batch.shape)
            bs.render(s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
        for i in range(bs.FRAMES):
            s1_frame = s1_batch[0, :, :, :, i].data.numpy()
            bs.render(s1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
        for i in range(bs.FRAMES):
            y1_frame = y1[0, :, :, :, i].data.numpy()
            bs.render(y1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))


if __name__ == "__main__":
    run_model_show_frames()
