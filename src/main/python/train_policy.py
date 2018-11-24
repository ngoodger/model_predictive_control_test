import os.path
from datetime import datetime, timedelta
import torch.distributed as dist

import block_dataset
import policy
import torch
from torch.utils.data import DataLoader
from google.cloud import storage
import json

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
MODEL_PATH = "my_model.pt"
MODEL_METADATA_PATH = "my_model_metadata.json"
POLICY_PATH = "my_policy.pt"
POLICY_METADATA_PATH = "my_policy_metadata.json"
PRINT_INTERVAL = 100
SAVE_INTERVAL = 100


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
    # Always load pre-trained model.
    model_bucket = os.environ["GCS_BUCKET"]
    client = storage.Client()
    bucket = client.get_bucket(model_bucket)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename(MODEL_PATH)
    model = torch.load(MODEL_PATH, map_location=device)

    policy_bucket = os.environ["GCS_BUCKET"]
    if POLICY_PATH in list_blob_names(model_bucket):
        print("Loading pre-trained policy.")
        client = storage.Client()
        bucket = client.get_bucket(policy_bucket)
        blob = bucket.blob(POLICY_PATH)
        blob.download_to_filename(POLICY_PATH)
        policy0 = torch.load(POLICY_PATH, map_location=device)
    else:
        # policy0 = torch.nn.DataParallel(policy_no_parallel).to(device)
        print("Starting from untrained policy.")
        policy_no_parallel = policy.Policy(
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
            device=device,
        )
        policy0 = policy_no_parallel.to(device)
    trainer = policy.PolicyTrainer(
        learning_rate=learning_rate, policy=policy0, model=model, world_size=world_size
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
            torch.save(policy0, POLICY_PATH)
    # return mean_loss
    metadata_dict = {
        "mean_loss": mean_loss,
        "training_time": (datetime.now() - start_train).total_seconds(),
    }
    json_metadata = json.dumps(metadata_dict)
    with open(POLICY_METADATA_PATH, "w") as f:
        f.write(json_metadata)
    return policy0


def list_blob_names(bucket_name):
    """Lists all the blob names in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob_name_list = [blob.name for blob in bucket.list_blobs()]
    return blob_name_list


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        # Assuming we are using a gpu
        space = {"learning_rate": 1e-3, "batch_size": 64, "world_size": world_size}
    else:
        # Assuming we are using a cpu
        space = {"learning_rate": 1e-4, "batch_size": 4, "world_size": world_size}
    policy0 = objective(space, timedelta(minutes=1))
    rank = dist.get_rank() if world_size > 1 else 0
    torch.save(policy0, POLICY_PATH)
    if rank == 0:
        policy_bucket = os.environ["GCS_BUCKET"]
        client = storage.Client()
        bucket = client.get_bucket(policy_bucket)
        blob = bucket.blob(POLICY_PATH)
        blob.upload_from_filename(POLICY_PATH)
        blob = bucket.blob(POLICY_METADATA_PATH)
        blob.upload_from_filename(POLICY_METADATA_PATH)
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
