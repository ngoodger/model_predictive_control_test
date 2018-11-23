import os.path
from datetime import datetime, timedelta

import block_dataset
import policy
import torch
from torch.utils.data import DataLoader

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)
POLICY_PATH = "my_policy.pt"
PRINT_INTERVAL = 100
SAVE_INTERVAL = 100


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    world_size = int(space["world_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dataset = block_dataset.PolicyDataSet(TRAINING_ITERATIONS)

    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    # Always load pre-trained model.
    model_bucket = os.environ["GCS_BUCKET"]
    client = storage.Client()
    bucket = client.get_bucket(model_bucket)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename(MODEL_PATH)
    model = torch.load("my_model.pt")

    if os.path.exists(POLICY_PATH):
        print("Loading pre-trained policy.")
        policy0 = torch.load(POLICY_PATH)
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
        trainer.train(
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
    return policy0


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    space = {"learning_rate": 1e-4, "batch_size": 1, "world_size": world_size}
    policy0 = objective(space, timedelta(hours=24))
    torch.save(policy0, POLICY_PATH)
    if rank == 0:
        policy_bucket = os.environ["GCS_BUCKET"]
        client = storage.Client()
        bucket = client.get_bucket(policy_bucket)
        blob = bucket.blob(POLICY_PATH)
        blob.upload_from_filename(POLICY_PATH)
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
