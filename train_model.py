import torch
from torch.utils.data import DataLoader
import block_sys
import block_sys as bs
import block_dataset
import model
import time
from datetime import datetime
from datetime import timedelta
import hyperopt
import pandas as pd
import math
import os
TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)


def objective(space, time_limit=TRAINING_TIME):
    force_add = space["force_add"]
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    samples_dataset = block_dataset.BlockDataSet(TRAINING_ITERATIONS)

    dataloader = DataLoader(samples_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    model0 = torch.nn.DataParallel(model.Model0(force_add=force_add)).to(device)
    trainer = model.Trainer(learning_rate=learning_rate, model=model0)
    iteration = 0
    start = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        s0_batch = data[0].to(device)
        force_batch = data[1].to(device)
        s1_batch = data[2].to(device)
        y1, mean_loss = trainer.train(s0_batch - 0.5, s1_batch,
                                      force_batch / block_sys.FORCE_SCALE)
        if iteration % 100 == 0:
            elapsed = datetime.now()
            elapsed = elapsed - start
            print("Time:" + str(elapsed))
            start = datetime.now()
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
        # Limit training time to TRAINING_TIME
        if datetime.now() - start_train > TRAINING_TIME:
            break
    return mean_loss


def tune_hyperparam():
    torch.multiprocessing.freeze_support()
    # Create the domain space
    space = {
            "force_add": hyperopt.hp.choice('force_add',[True, False]),
            "batch_size": hyperopt.hp.qloguniform('batch_size', math.log(16), math.log(512), 16),
            "learning_rate":   hyperopt.hp.loguniform('learning_rate', math.log(1e-4), math.log(1e-1)),
            }
    # Create the algorithm
    tpe_algo = hyperopt.tpe.suggest
    # Create a trials object
    tpe_trials = hyperopt.Trials()
    tpe_best = hyperopt.fmin(fn=objective, space=space,
                             algo=tpe_algo, trials=tpe_trials,
                             max_evals=20)
    print(tpe_best)

    tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results],
                                'force_add': tpe_trials.idxs_vals[1]['force_add'],
                                'learning_rate': tpe_trials.idxs_vals[1]['learning_rate'],
                                'batch_size': tpe_trials.idxs_vals[1]['batch_size'],
                                })

    print(tpe_results.head(20))


def train_model():
    space = {"force_add": True, "learning_rate": 1e-3, "batch_size": 128}
    objective(space, timedelta(hours=1))
