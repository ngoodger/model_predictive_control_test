import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import block_sys
import block_sys as bs
import model
import random
import time
import numpy as np
BATCH_SIZE = 16
EPOCHS = 5
TRAINING_DATA_SIZE = 1000
IMAGE_DEPTH = 1

class BlockDataSet(Dataset):
    def __init__(self):
        #super(BlockDataSet, self).__init__()
        self.my_block_sys = bs.BlockSys()
        self.s0 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                      dtype=np.float32)
        self.s1 = np.zeros([IMAGE_DEPTH, bs.GRID_SIZE, 4 * bs.GRID_SIZE],
                      dtype=np.float32)
        self.force = np.zeros([2], dtype=np.float32)

    def __len__(self):
        return TRAINING_DATA_SIZE

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


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    s0 = pickle.load(open("s0.p", "rb"))
    force = pickle.load(open("force.p", "rb"))
    s1 = pickle.load(open("s1.p", "rb"))

    s0_tensor = torch.from_numpy(s0)
    force_tensor = torch.from_numpy(force)
    s1_tensor = torch.from_numpy(s1)
    #samples_dataset = torch.utils.data.TensorDataset(s0_tensor,
    #                                                 force_tensor, s1_tensor)

    samples_dataset = BlockDataSet()
    print("samples" + str(len(samples_dataset)))

    dataloader = DataLoader(samples_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    cnn_model = torch.nn.DataParallel(model.ConvNet()).to(device)
    trainer = model.Trainer(cnn_model)
    iteration = 0
    start = time.clock()
    for epoch_idx in range(EPOCHS):
        print("epoch: {}".format(epoch_idx))
        for batch_idx, data in enumerate(dataloader):
            s0_batch = data[0].to(device)
            force_batch = data[1].to(device)
            s1_batch = data[2].to(device)
            y1 = trainer.train(s0_batch - 0.5, s1_batch,
                               force_batch / block_sys.FORCE_SCALE)
            if iteration % 100 == 0:
                elapsed = time.clock()
                elapsed = elapsed - start
                print("Time:" + str(elapsed))
                start = time.clock()
                for i in range(4):
                    time.sleep(0.1)
                    s0_frame = s0_batch[0, :, :, (i * bs.GRID_SIZE):
                                  (i * bs.GRID_SIZE + bs.GRID_SIZE)].cpu().numpy()
                    block_sys.render(s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))

                for i in range(4):
                    time.sleep(0.1)
                    y1_frame = y1[0, :, :, (i * bs.GRID_SIZE):
                                  (i * bs.GRID_SIZE + bs.GRID_SIZE)].cpu().numpy()
                    block_sys.render(y1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
            iteration += 1
