from datetime import datetime, timedelta

import block_dataset
import policy
import torch
from torch.utils.data import DataLoader

TRAINING_ITERATIONS = 100000000
TRAINING_TIME = timedelta(minutes=20)


def objective(space, time_limit=TRAINING_TIME):
    learning_rate = space["learning_rate"]
    batch_size = int(space["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dataset = block_dataset.PolicyDataSet(TRAINING_ITERATIONS)

    dataloader = DataLoader(
        samples_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    model = torch.load("my_model.pt")
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
    )
    policy0 = torch.nn.DataParallel(policy_no_parallel).to(device)
    trainer = policy.PolicyTrainer(
        learning_rate=learning_rate, policy=policy0, model=model
    )
    iteration = 0
    start = datetime.now()
    start_train = datetime.now()
    for batch_idx, data in enumerate(dataloader):
        force_0_batch = data[0].to(device)
        s0_batch = data[1].to(device)
        s1_batch = data[2].to(device)
        trainer.train({"s0": s0_batch, "s1": s1_batch, "force_0": force_0_batch})
        # print(y1)
        if iteration % 1000 == 0:
            elapsed = datetime.now()
            elapsed = elapsed - start
            print("Time:" + str(elapsed))
            start = datetime.now()
        iteration += 1
        # Limit training time to TRAINING_TIME
        if datetime.now() - start_train > time_limit:
            break
    # return mean_loss
    return policy0


if __name__ == "__main__":
    space = {"learning_rate": 1e-3, "batch_size": 16}
    my_policy = objective(space, timedelta(hours=6))
    torch.save(my_policy, "my_policy.pt")
    # model = torch.load('my_model.pt')

    # .. to load your previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))
