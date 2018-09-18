import math
from datetime import datetime, timedelta

# import block_sys
# import block_sys as bs
import block_dataset
import hyperopt
import model
import pandas as pd
import torch
from torch.utils.data import DataLoader

# import os

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dataset = block_dataset.BlockDataSet(TRAINING_ITERATIONS)

    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    model_no_parallel = model.Model(
        layer_1_cnn_filters=16,
        layer_2_cnn_filters=16,
        layer_3_cnn_filters=16,
        layer_4_cnn_filters=32,
        layer_1_kernel_size=3,
        layer_2_kernel_size=3,
        layer_3_kernel_size=3,
        layer_4_kernel_size=3,
        force_hidden_layer_size=32,
        middle_hidden_layer_size=128,
    )
    model0 = torch.nn.DataParallel(model_no_parallel).to(device)
    trainer = model.ModelTrainer(learning_rate=learning_rate, model=model0)
    iteration = 0
    # start = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        force_0_batch = data[0].to(device)
        # print(force_0_batch)
        s0_batch = data[1].to(device)
        # print(s0_batch)
        force_1_batch = data[2].to(device)
        # print(force_1_batch)
        s1_batch = data[3].to(device)
        # print(s1_batch)
        # y1, mean_loss = trainer.train({"s0":s0_batch, "s1":s1_batch, "force_0":force_0_batch, "force_1": force_1_batch})
        trainer.train(
            {
                "s0": s0_batch,
                "s1": s1_batch,
                "force_0": force_0_batch,
                "force_1": force_1_batch,
            }
        )
        # print(y1)
        """
        if iteration % 1000 == 0:
            elapsed = datetime.now()
            elapsed = elapsed - start
            print("Time:" + str(elapsed))
            start = datetime.now()
            if not os.name == "nt":
                for i in range(4):
                    # time.sleep(0.1)
                    s0_frame = s0_batch[0, :, :, :, i].data.numpy()
                    print(s0_batch.shape)
                    block_sys.render(s0_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))

                for i in range(bs.FRAMES):
                    # time.sleep(0.1)
                    s1_frame = s1_batch[0, :, :, :, i].data.numpy()
                    block_sys.render(s1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
                for i in range(bs.FRAMES):
                    # time.sleep(0.1)
                    y1_frame = y1[0, :, :, :, i].data.numpy()
                    block_sys.render(y1_frame.reshape([bs.GRID_SIZE, bs.GRID_SIZE]))
        """
        iteration += 1
        # Limit training time to TRAINING_TIME
        if datetime.now() - start_train > time_limit:
            break
    # return mean_loss
    return model0


def tune_hyperparam():
    torch.multiprocessing.freeze_support()
    # Create the domain space
    space = {
        "batch_size": hyperopt.hp.qloguniform(
            "batch_size", math.log(16), math.log(512), 16
        ),
        "learning_rate": hyperopt.hp.loguniform(
            "learning_rate", math.log(1e-4), math.log(1e-1)
        ),
    }
    # Create the algorithm
    tpe_algo = hyperopt.tpe.suggest
    # Create a trials object
    tpe_trials = hyperopt.Trials()
    tpe_best = hyperopt.fmin(
        fn=objective, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=20
    )
    print(tpe_best)

    tpe_results = pd.DataFrame(
        {
            "loss": [x["loss"] for x in tpe_trials.results],
            "learning_rate": tpe_trials.idxs_vals[1]["learning_rate"],
            "batch_size": tpe_trials.idxs_vals[1]["batch_size"],
        }
    )

    print(tpe_results.head(20))


if __name__ == "__main__":
    space = {"learning_rate": 3e-3, "batch_size": 16}
    my_model = objective(space, timedelta(hours=6))
    torch.save(my_model, "my_model.pt")
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
