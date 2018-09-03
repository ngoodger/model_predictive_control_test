import pickle
import torch
from torch.utils.data import DataLoader
import block_sys
import block_sys as bs
import model
import time
BATCH_SIZE = 32
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


s0 = pickle.load(open("s0.p", "rb"))
force = pickle.load(open("force.p", "rb"))
s1 = pickle.load(open("s1.p", "rb"))

s0_tensor = torch.from_numpy(s0)
force_tensor = torch.from_numpy(force)
s1_tensor = torch.from_numpy(s1)
samples_dataset = torch.utils.data.TensorDataset(s0_tensor,
                                                 force_tensor, s1_tensor)

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
