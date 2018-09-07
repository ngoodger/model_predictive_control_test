import torch
from torch.utils.data import DataLoader
import block_sys
import block_sys as bs
import block_dataset
import model
import time
import hyperopt
import pandas as pd
import math
import os
BATCH_SIZE = 32
TRAINING_ITERATIONS = 1000000


def objective(learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    """
    s0 = pickle.load(open("s0.p", "rb"))
    force = pickle.load(open("force.p", "rb"))
    s1 = pickle.load(open("s1.p", "rb"))

    s0_tensor = torch.from_numpy(s0)
    force_tensor = torch.from_numpy(force)
    s1_tensor = torch.from_numpy(s1)
    """
    #samples_dataset = torch.utils.data.TensorDataset(s0_tensor,
    #                                                 force_tensor, s1_tensor)

    samples_dataset = block_dataset.BlockDataSet(TRAINING_ITERATIONS)
    print("samples" + str(len(samples_dataset)))

    dataloader = DataLoader(samples_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    cnn_model = torch.nn.DataParallel(model.ConvNet()).to(device)
    trainer = model.Trainer(learning_rate=learning_rate, cnn_model=cnn_model)
    iteration = 0
    start = time.clock()
    for batch_idx, data in enumerate(dataloader):
        s0_batch = data[0].to(device)
        force_batch = data[1].to(device)
        s1_batch = data[2].to(device)
        y1, mean_loss = trainer.train(s0_batch - 0.5, s1_batch,
                                      force_batch / block_sys.FORCE_SCALE)
        if iteration % 100 == 0:
            elapsed = time.clock()
            elapsed = elapsed - start
            print("Time:" + str(elapsed))
            start = time.clock()
            if not os.name == "nt":
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
    return mean_loss


def tune_hyperparam():
    torch.multiprocessing.freeze_support()
    # Create the domain space
    learning_rate= hyperopt.hp.loguniform('learning_rate', math.log(1e-4), math.log(1e-1))
    # Create the algorithm
    tpe_algo = hyperopt.tpe.suggest
    # Create a trials object
    tpe_trials = hyperopt.Trials()
    tpe_best = hyperopt.fmin(fn=objective, space=learning_rate,
                             algo=tpe_algo, trials=tpe_trials,
                             max_evals=10)
    print(tpe_best)

    tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                'iteration': tpe_trials.idxs_vals[0]['learning_rate'],
                                'x': tpe_trials.idxs_vals[1]['learning_rate']})

    print(tpe_results.head(10))
