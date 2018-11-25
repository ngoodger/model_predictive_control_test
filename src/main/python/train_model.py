# import os.path
from datetime import datetime, timedelta
import blob_handler


# from tensorboardX import SummaryWriter
import torch.distributed as dist

import json
import input_cnn

# import block_sys # import block_sys as bs
import block_dataset
import model
import os

# import pandas as pd
import torch
from torch.utils.data import DataLoader

# import os

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
INPUT_CNN_PATH = "input_cnn.pt"
MODEL_PATH = "recurrent_model.pt"
MODEL_METADATA_PATH = "recurrent_model_metadata.json"
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
    my_blob_handler = blob_handler.BlobHandler(os.environ["GCS_BUCKET"])
    if MODEL_PATH in my_blob_handler.ls_blob():
        print("Loading pre-existing recurrent model.")
        my_blob_handler.download_blob(MODEL_PATH)
        model0 = torch.load(MODEL_PATH, map_location=device)
    else:
        print("Starting from untrained recurrent model.")
        model_no_parallel = model.RecurrentModel(
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
    if INPUT_CNN_PATH in my_blob_handler.ls_blob():
        print("Loading pre-existing input cnn")
        my_blob_handler.download_blob(INPUT_CNN_PATH)
        my_input_cnn = torch.load(INPUT_CNN_PATH, map_location=device)
    else:
        print("Starting from untrained input cnn.")
        my_input_cnn = input_cnn.InputCNN(
            layer_1_cnn_filters=16,
            layer_2_cnn_filters=16,
            layer_3_cnn_filters=16,
            layer_4_cnn_filters=32,
            layer_1_kernel_size=3,
            layer_2_kernel_size=3,
            layer_3_kernel_size=3,
            layer_4_kernel_size=3,
        )

    trainer = model.RecurrentModelTrainer(
        learning_rate=learning_rate,
        input_cnn=my_input_cnn,
        model=model0,
        world_size=world_size,
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
        mean_loss = trainer.train(batch_data)
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
            torch.save(my_input_cnn, INPUT_CNN_PATH)
    metadata_dict = {
        "mean_loss": mean_loss,
        "training_time": (datetime.now() - start_train).total_seconds(),
    }
    json_metadata = json.dumps(metadata_dict)
    rank = dist.get_rank() if world_size > 1 else 0
    with open(MODEL_METADATA_PATH, "w") as f:
        f.write(json_metadata)
    torch.save(model0, MODEL_PATH)
    torch.save(my_input_cnn, INPUT_CNN_PATH)
    # From master save to storage bucket.
    if rank == 0:
        print("Saving model to storage bucket")
        my_blob_handler.upload_blob(INPUT_CNN_PATH)
        my_blob_handler.upload_blob(MODEL_PATH)
        my_blob_handler.upload_blob(MODEL_METADATA_PATH)


if __name__ == "__main__":
    # Only use distributed data parallel if world_size > 1.
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        # If cuda is available we assume that we are using it.
        if torch.cuda.is_available():
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("tcp")
    if torch.cuda.is_available():
        # Assuming we are using a gpu
        space = {"learning_rate": 1e-3, "batch_size": 64, "world_size": world_size}
    else:
        # Assuming we are using a cpu
        space = {"learning_rate": 1e-4, "batch_size": 4, "world_size": world_size}
    objective(space, timedelta(minutes=20))
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # .load_state_dict(torch.load('mytraining.pt'))
