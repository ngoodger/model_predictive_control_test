# import os.path
from datetime import datetime, timedelta

# from tensorboardX import SummaryWriter
import torch.distributed as dist

# import block_sys # import block_sys as bs
import block_dataset
import model
import os
from google.cloud import storage

# import pandas as pd
import torch
from torch.utils.data import DataLoader

# import os

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
MODEL_PATH = "my_model.pt"
SEQ_LEN = 4
SAVE_INTERVAL = 100
PRINT_INTERVAL = 100


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    world_size = int(space["world_size"])
    # writer = SummaryWriter("log_files/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if world_size > 1 else 0
    samples_dataset = block_dataset.ModelDataSet(TRAINING_ITERATIONS, SEQ_LEN, rank)

    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if os.path.exists(MODEL_PATH):
        print("Loading pre-existing model.")
        model0 = torch.load(MODEL_PATH)
    else:
        print("Starting from untrained model.")
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
            recurrent_layer_size=128,
            device=device,
        )
        model0 = model_no_parallel.to(device)

    trainer = model.ModelTrainer(
        learning_rate=learning_rate, model=model0, world_size=world_size
    )
    iteration = 0
    start = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        forces, observations = data
        forces_device = [torch.tensor(force, device=device) for force in forces]
        observations_device = [
            torch.tensor(observation, device=device) for observation in observations
        ]
        batch_data = {
            "forces": forces_device,
            "observations": observations_device,
            "seq_len": SEQ_LEN,
        }
        trainer.train(batch_data)
        if iteration % PRINT_INTERVAL == 0 and rank == 0:
            # writer.add_scalar("Train/Loss", loss, batch_idx)
            elapsed = datetime.now()
            elapsed = elapsed - start
            print(
                "Samples / Sec: {}".format(
                    (world_size * PRINT_INTERVAL * batch_size) / elapsed.total_seconds()
                )
            )
            print("Time:" + str(elapsed))
            start = datetime.now()
        iteration += 1
        # Limit training time to TRAINING_TIME
        if datetime.now() - start_train > time_limit:
            break
        if iteration % SAVE_INTERVAL == 0:
            torch.save(model0, MODEL_PATH)
    return model0


if __name__ == "__main__":
    # Only use distributed data parallel if world_size > 1.
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        # If cuda is available we assume that we are using it.
        if torch.cuda.is_available():
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("tcp")
    space = {"learning_rate": 1e-3, "batch_size": 4, "world_size": world_size}
    model0 = objective(space, timedelta(seconds=30))
    rank = dist.get_rank() if world_size > 1 else 0
    torch.save(model0, MODEL_PATH)
    # On master save to storage bucket.
    if rank == 0:
        model_bucket = os.environ["GCS_BUCKET"]
        client = storage.Client()
        bucket = client.get_bucket(model_bucket)
        blob = bucket.blob(MODEL_PATH)
        blob.upload_from_filename(MODEL_PATH)
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
