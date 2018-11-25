import torch
import trainer
from block_sys import GRID_SIZE, IMAGE_DEPTH, FORCE_SCALE
from torch import nn

# from model import forward_sequence

LOSS_MEAN_WINDOW = 100000
PRINT_LOSS_MEAN_ITERATION = 100

STRIDE = 2
TARGET_HORIZON = 3


class PolicyTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, policy, model):
        super(PolicyTrainer, self).__init__(learning_rate, policy)
        self.model = model

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_loss(self, batch_data):
        for i in range(TARGET_HORIZON):
            force_1 = self.nn_module.forward(batch_data)
            # Augment batch data with force from policy
            batch_data["force_1"] = force_1 * FORCE_SCALE * 0.5
            logits, out = self.model.forward(batch_data)
            # Loss is only relative to the final frame.
            # We just want zero velocity at the goal.  We don't care how we get there.
            logits_last_frame = logits[:, :, :, :, 3]
        y = batch_data["target"]
        logits_flat = logits_last_frame.reshape([logits_last_frame.size(0), -1])
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
    ):
        """
        force_add determines whether the force is added or concatonated.
        """

        self.layer_1_cnn_filters = layer_1_cnn_filters
        self.layer_2_cnn_filters = layer_2_cnn_filters
        self.layer_3_cnn_filters = layer_3_cnn_filters
        self.layer_4_cnn_filters = layer_4_cnn_filters
        self.layer_1_kernel_size = layer_1_kernel_size
        self.layer_2_kernel_size = layer_2_kernel_size
        self.layer_3_kernel_size = layer_3_kernel_size
        self.layer_4_kernel_size = layer_4_kernel_size
        self.force_hidden_layer_size = force_hidden_layer_size
        self.middle_hidden_layer_size = middle_hidden_layer_size
        LAYERS = 4
        self.middle_layer_image_width = int(GRID_SIZE / (2 ** (LAYERS - 1)))
        middle_layer_size = int(
            layer_4_cnn_filters * 1 * (self.middle_layer_image_width ** 2)
        )
        self.middle_layer_size = middle_layer_size
        print(self.middle_layer_image_width)
        print(self.middle_layer_size)
        super(Policy, self).__init__()
        self.layer1 = nn.Sequential(
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
        self.layer2 = nn.Sequential(
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
        self.layer3 = nn.Sequential(
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
        self.layer4 = nn.Sequential(
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
        self.layer_force_0 = nn.Sequential(
            nn.Linear(2, middle_layer_size), nn.LeakyReLU()
        )

        self.layer_middle_0 = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_middle_1 = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_middle_2 = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, 2), nn.Tanh()
        )

    def forward(self, batch_data):
        x = batch_data["s0"]
        x_force_0 = batch_data["force_0"]
        print_shape = False
        out_force = self.layer_force_0(x_force_0)
        out1 = self.layer1(x)
        if print_shape:
            print(out1.shape)
        out2 = self.layer2(out1)
        if print_shape:
            print(out2.shape)
        out3 = self.layer3(out2)
        if print_shape:
            print(out3.shape)
        out4 = self.layer4(out3)
        if print_shape:
            print(out4.shape)
        out4_flat = out4.view(out4.size(0), -1)
        out_middle_0 = self.layer_middle_0(torch.add(out4_flat, out_force))
        # Add block force.
        out_combined = self.layer_middle_1(out_middle_0)
        out = self.layer_middle_2(out_combined)
        return out
