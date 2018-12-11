import os.path
from datetime import datetime, timedelta
import torch.distributed as dist
import blob_handler
import input_cnn

import block_dataset
import policy
import torch
from torch.utils.data import DataLoader
import json

USE_POLICY_SPECIFIC_INPUT_CNN = False
TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
POLICY_INPUT_CNN_PATH = "policy_input_cnn.pt"
MODEL_INPUT_CNN_PATH = "input_cnn.pt"
MODEL_PATH = "recurrent_model.pt"
MODEL_METADATA_PATH = "my_model_metadata.json"
POLICY_PATH = "my_policy.pt"
POLICY_METADATA_PATH = "my_policy_metadata.json"
PRINT_INTERVAL = 100
SAVE_INTERVAL = 1000


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    world_size = int(space["world_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = dist.get_rank() if world_size > 1 else 0
    samples_dataset = block_dataset.PolicyDataSet(TRAINING_ITERATIONS, rank)

    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    # Always load pre-trained recurrent model and input_cnn
    my_blob_handler = blob_handler.BlobHandler(os.environ["GCS_BUCKET"])
    my_blob_handler.download_blob(MODEL_PATH)
    model = torch.load(MODEL_PATH, map_location=device)
    my_blob_handler.download_blob(MODEL_INPUT_CNN_PATH)
    my_input_cnn_model = torch.load(MODEL_INPUT_CNN_PATH, map_location=device)

    if POLICY_PATH in my_blob_handler.ls_blob():
        print("Loading pre-trained policy.")
        my_blob_handler.download_blob(POLICY_PATH)
        policy0 = torch.load(POLICY_PATH, map_location=device)
    else:
        # policy0 = torch.nn.DataParallel(policy_no_parallel).to(device)
        print("Starting from untrained policy.")
        policy_no_parallel = policy.Policy(
            force_hidden_layer_size=32, middle_hidden_layer_size=128, device=device
        )
        policy0 = policy_no_parallel.to(device)
    if USE_POLICY_SPECIFIC_INPUT_CNN:
        if POLICY_INPUT_CNN_PATH in my_blob_handler.ls_blob():
            print("Loading pre-existing input cnn")
            my_blob_handler.download_blob(POLICY_INPUT_CNN_PATH)
            my_input_cnn = torch.load(POLICY_INPUT_CNN_PATH, map_location=device)
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
    else:
        my_blob_handler.download_blob(MODEL_INPUT_CNN_PATH)
        my_input_cnn = torch.load(MODEL_INPUT_CNN_PATH, map_location=device)

    trainer = policy.PolicyTrainer(
        learning_rate=learning_rate,
        input_cnn=my_input_cnn,
        policy=policy0,
        model_input_cnn=my_input_cnn_model,
        model=model,
        world_size=world_size,
        train_input_cnn=USE_POLICY_SPECIFIC_INPUT_CNN,
    )
    iteration = 0
    start_time = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        force_0, start, target = data
        force_0_device = torch.tensor(force_0, device=device)
        start_device = torch.tensor(start, device=device)
        target_device = torch.tensor(target, device=device)
        mean_loss = trainer.train(
            {"start": start_device, "target": target_device, "force_0": force_0_device}
        )
        if iteration % PRINT_INTERVAL == 0:
            elapsed = datetime.now()
            elapsed = elapsed - start_time
            print(
                "Samples / Sec: {}".format(
                    (world_size * PRINT_INTERVAL * batch_size) / elapsed.total_seconds()
                )
            )
            print("Time:" + str(elapsed))
            start_time = datetime.now()
        iteration += 1
        # Limit training time to TRAINING_TIME
        if datetime.now() - start_train > time_limit:
            break
        if iteration % SAVE_INTERVAL == 0:
            metadata_dict = {
                "mean_loss": mean_loss,
                "training_time": (datetime.now() - start_train).total_seconds(),
            }
            json_metadata = json.dumps(metadata_dict)
            rank = dist.get_rank() if world_size > 1 else 0
            with open(POLICY_METADATA_PATH, "w") as f:
                f.write(json_metadata)
            if USE_POLICY_SPECIFIC_INPUT_CNN:
                torch.save(my_input_cnn, POLICY_INPUT_CNN_PATH)
                my_blob_handler.upload_blob(POLICY_INPUT_CNN_PATH)
            torch.save(policy0, POLICY_PATH)
            my_blob_handler.upload_blob(POLICY_PATH)
            my_blob_handler.upload_blob(POLICY_METADATA_PATH)
    # return mean_loss
    metadata_dict = {
        "mean_loss": mean_loss,
        "training_time": (datetime.now() - start_train).total_seconds(),
    }
    json_metadata = json.dumps(metadata_dict)
    rank = dist.get_rank() if world_size > 1 else 0
    with open(POLICY_METADATA_PATH, "w") as f:
        f.write(json_metadata)
    if USE_POLICY_SPECIFIC_INPUT_CNN:
        torch.save(my_input_cnn, POLICY_INPUT_CNN_PATH)
        my_blob_handler.upload_blob(POLICY_INPUT_CNN_PATH)
    torch.save(policy0, POLICY_PATH)
    if rank == 0:
        my_blob_handler.upload_blob(POLICY_PATH)
        my_blob_handler.upload_blob(POLICY_METADATA_PATH)


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size > 1:
        # If cuda is available we assume that we are using it.
        if torch.cuda.is_available():
            dist.init_process_group("nccl")
        else:
            dist.init_process_group("tcp")
    if torch.cuda.is_available():
        # Assuming we are using a gpu
        space = {"learning_rate": 1e-4, "batch_size": 32, "world_size": world_size}
    else:
        # Assuming we are using a cpu
        space = {"learning_rate": 1e-5, "batch_size": 4, "world_size": world_size}
    objective(space, timedelta(hours=1))
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
