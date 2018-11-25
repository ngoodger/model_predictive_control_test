import torch
import trainer
from block_sys import GRID_SIZE, FORCE_SCALE
from torch import nn

TARGET_HORIZON = 3
STRIDE = 2


class PolicyTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, input_cnn, policy, model, world_size):
        # We only train policy here.
        # model and input_cnn must be trained by training model.
        parameters = policy.parameters()
        super(PolicyTrainer, self).__init__(learning_rate, parameters, world_size)
        self.input_cnn = input_cnn
        self.policy = policy
        self.model = model

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_loss(self, batch_data):
        force_0 = batch_data["force_0"]
        start = batch_data["start"]
        y = batch_data["target"]
        out_target_cnn_flat = self.input_cnn.forward(y)
        for i in range(TARGET_HORIZON):
            if i == 0:
                out_start_cnn_flat = self.input_cnn.forward(start)
                force_1_unscaled, out_target_cnn_layer = self.policy.forward(
                    force_0,
                    out_start_cnn_flat,
                    out_target_cnn_flat,
                    out_target_cnn_layer=None,
                    first_iteration=True,
                )
                # force_1 = force_1_unscaled * FORCE_SCALE * 0.5
                force_1 = force_1_unscaled
                logits, out, recurrent_state = self.model.forward(
                    out_start_cnn_flat, None, force_0, force_1, first_iteration=True
                )
            else:
                out_start_cnn_flat = self.input_cnn.forward(out)
                force_1_unscaled, _ = self.policy.forward(
                    force_0,
                    out_start_cnn_flat,
                    out_target_cnn_flat,
                    out_target_cnn_layer=out_target_cnn_layer,
                    first_iteration=False,
                )
                # force_1 = force_1_unscaled * FORCE_SCALE * 0.5
                force_1 = force_1_unscaled
                logits, out, recurrent_state = self.model.forward(
                    out_start_cnn_flat,
                    recurrent_state,
                    force_0,
                    force_1,
                    first_iteration=False,
                )
            force_0 = force_1
        logits_flat = logits.reshape([logits.size(0), -1])
        y_flat = y.reshape([y.size(0), -1])
        loss = self.criterion(logits_flat, y_flat)
        return loss


class Policy(nn.Module):
    def __init__(self, force_hidden_layer_size, middle_hidden_layer_size, device):
        """
        force_add determines whether the force is added or concatonated.
        """

        super(Policy, self).__init__()
        self.middle_hidden_layer_size = middle_hidden_layer_size
        self.device = device
        layer_4_cnn_filters = 32
        LAYERS = 4
        self.middle_layer_image_width = int(GRID_SIZE / (2 ** (LAYERS - 1)))
        middle_layer_size = int(
            layer_4_cnn_filters * 1 * (self.middle_layer_image_width ** 2)
        )
        self.middle_layer_size = middle_layer_size
        self.layer_force = nn.Sequential(
            nn.Linear(2, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_start_cnn = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_target_cnn = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_policy_hidden = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_policy = nn.Linear(middle_hidden_layer_size, 2)

    def forward(
        self,
        force_0,
        out_start_cnn_flat,
        out_target_cnn_flat,
        out_target_cnn_layer,
        first_iteration=False,
    ):

        batch_size = force_0.size(0)
        out_force = self.layer_force(force_0)

        # Feed both the start and target cnn through 1 layer to ensure
        # the model can place them in different parts of the input vector.
        out_layer_start_cnn = self.layer_start_cnn(out_start_cnn_flat)
        if first_iteration:
            out_target_cnn_layer = self.layer_target_cnn(out_target_cnn_flat)
        # Combine outputs from cnn and force layers
        combined = torch.add(
            out_target_cnn_layer, torch.add(out_layer_start_cnn, out_force)
        )
        out_policy_hidden = self.layer_policy_hidden(combined.view(batch_size, -1))
        out_policy = self.layer_policy(out_policy_hidden)
        return out_policy, out_target_cnn_layer
