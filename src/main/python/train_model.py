import math
import os.path
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
MODEL_PATH = "my_model.pt"
SEQ_LEN = 10


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dataset = block_dataset.ModelDataSet(TRAINING_ITERATIONS, SEQ_LEN)

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
        recurrent_layer_size=32,
    )
    if os.path.exists(MODEL_PATH):
        model0 = torch.load(MODEL_PATH)
    else:
        model0 = torch.nn.DataParallel(model_no_parallel).to(device)
    trainer = model.ModelTrainer(learning_rate=learning_rate, model=model0)
    iteration = 0
    start = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        force = data[0]
        s = data[1]
        trainer.train({"s": s, "force": force, "seq_len": SEQ_LEN})
        if iteration % 100 == 0:
            elapsed = datetime.now()
            elapsed = elapsed - start
            print("Time:" + str(elapsed))
            start = datetime.now()
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
    space = {"learning_rate": 1e-4, "batch_size": 1}
    my_model = objective(space, timedelta(hours=1))
    torch.save(my_model, MODEL_PATH)
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
