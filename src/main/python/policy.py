import torch
import trainer
from block_sys import GRID_SIZE, IMAGE_DEPTH, FORCE_SCALE
from torch import nn

TARGET_HORIZON = 3
STRIDE = 2


class PolicyTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, policy, model, world_size):
        super(PolicyTrainer, self).__init__(learning_rate, policy, world_size)
        self.model = model

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_loss(self, batch_data):
        force_0 = batch_data["force_0"]
        start = batch_data["start"]
        y = batch_data["target"]
        for i in range(TARGET_HORIZON):
            if i == 0:
                force_1 = self.nn_module.forward(force_0, start) * FORCE_SCALE * 0.5
                logits, out, recurrent_state = self.model.forward(
                    start, None, force_0, force_1, first_iteration=True
                )
            else:
                force_1 = self.nn_module.forward(force_0, out) * FORCE_SCALE * 0.5
                logits, out, recurrent_state = self.model.forward(
                    out, recurrent_state, force_0, force_1, first_iteration=False
                )
            force_0 = force_1
        logits_flat = logits.reshape([logits.size(0), -1])
        y_flat = y.reshape([y.size(0), -1])
        loss = self.criterion(logits_flat, y_flat)
        return loss


class Policy(nn.Module):
    def __init__(
        self,
        layer_1_cnn_filters,
        layer_2_cnn_filters,
        layer_3_cnn_filters,
        layer_4_cnn_filters,
        layer_1_kernel_size,
        layer_2_kernel_size,
        layer_3_kernel_size,
        layer_4_kernel_size,
        force_hidden_layer_size,
        middle_hidden_layer_size,
        device,
    ):
        """
        force_add determines whether the force is added or concatonated.
        """

        super(Policy, self).__init__()
        self.layer_1_cnn_filters = layer_1_cnn_filters
        self.layer_2_cnn_filters = layer_2_cnn_filters
        self.layer_3_cnn_filters = layer_3_cnn_filters
        self.layer_4_cnn_filters = layer_4_cnn_filters
        self.layer_1_kernel_size = layer_1_kernel_size
        self.layer_2_kernel_size = layer_2_kernel_size
        self.layer_3_kernel_size = layer_3_kernel_size
        self.layer_4_kernel_size = layer_4_kernel_size
        self.middle_hidden_layer_size = middle_hidden_layer_size
        self.device = device

        LAYERS = 4
        self.middle_layer_image_width = int(GRID_SIZE / (2 ** (LAYERS - 1)))
        middle_layer_size = int(
            layer_4_cnn_filters * 1 * (self.middle_layer_image_width ** 2)
        )
        self.middle_layer_size = middle_layer_size
        self.layer_cnn_0 = nn.Sequential(
            nn.Conv3d(
                IMAGE_DEPTH,
                layer_1_cnn_filters,
                kernel_size=layer_1_kernel_size,
                stride=[1, 1, 1],
                padding=int(layer_1_kernel_size / 2),
            ),
            # nn.BatchNorm2d(2),
            nn.LeakyReLU(),
        )
        self.layer_cnn_1 = nn.Sequential(
            nn.Conv3d(
                layer_1_cnn_filters,
                layer_2_cnn_filters,
                kernel_size=layer_2_kernel_size,
                stride=[STRIDE, STRIDE, 1],
                padding=int(layer_2_kernel_size / 2),
            ),
            # nn.BatchNorm2d(4),
            nn.LeakyReLU(),
        )
        self.layer_cnn_2 = nn.Sequential(
            nn.Conv3d(
                layer_2_cnn_filters,
                layer_3_cnn_filters,
                kernel_size=layer_3_kernel_size,
                stride=[STRIDE, STRIDE, STRIDE],
                padding=int(layer_3_kernel_size / 2),
            ),
            # nn.BatchNorm2d(4),
            nn.LeakyReLU(),
        )
        self.layer_cnn_3 = nn.Sequential(
            nn.Conv3d(
                layer_3_cnn_filters,
                layer_4_cnn_filters,
                kernel_size=layer_4_kernel_size,
                stride=[STRIDE, STRIDE, STRIDE],
                padding=int(layer_4_kernel_size / 2),
            ),
            # nn.BatchNorm2d(4),
            nn.LeakyReLU(),
        )
        self.layer_tcnn_0 = nn.Sequential(
            nn.ConvTranspose3d(
                layer_4_cnn_filters,
                layer_3_cnn_filters,
                kernel_size=layer_4_kernel_size,
                stride=[STRIDE, STRIDE, STRIDE],
                padding=int(layer_4_kernel_size / 3),
                output_padding=[1, 1, 1],
            ),
            # nn.BatchNorm2d(4),
            nn.LeakyReLU(),
        )
        self.layer_force = nn.Sequential(
            nn.Linear(2, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_cnn = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_policy_hidden = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_policy = nn.Linear(middle_hidden_layer_size, 2)

    def forward(self, force_0, start, first_iteration=False):

        batch_size = force_0.size(0)
        out_force = self.layer_force(force_0)

        out_cnn_0 = self.layer_cnn_0(start)
        out_cnn_1 = self.layer_cnn_1(out_cnn_0)
        out_cnn_2 = self.layer_cnn_2(out_cnn_1)
        out_cnn_3 = self.layer_cnn_3(out_cnn_2)
        out_input_image_flat = out_cnn_3.view(out_cnn_3.size(0), -1)
        out_cnn = self.layer_cnn(out_input_image_flat)

        # Combine outputs from cnn and force layers
        combined = torch.add(out_cnn, out_force)
        out_policy_hidden = self.layer_policy_hidden(combined.view(batch_size, -1))
        out_policy = self.layer_policy(out_policy_hidden)
        return out_policy
