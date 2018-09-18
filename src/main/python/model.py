import torch
import trainer
from block_sys import GRID_SIZE, IMAGE_DEPTH
from torch import nn

STRIDE = 2


class ModelTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, model):
        super(ModelTrainer, self).__init__(learning_rate, model)

    def criterion(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def calc_loss(self, batch_data):
        logits, out = self.model.forward(batch_data)
        y = batch_data["s1"]
        loss = self.criterion(
            logits.reshape([logits.size(0), -1]), y.reshape([y.size(0), -1])
        )
        return loss


class Model(nn.Module):
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
        self.middle_hidden_layer_size = middle_hidden_layer_size
        LAYERS = 4
        self.middle_layer_image_width = int(GRID_SIZE / (2 ** (LAYERS - 1)))
        middle_layer_size = int(
            layer_4_cnn_filters * 1 * (self.middle_layer_image_width ** 2)
        )
        self.middle_layer_size = middle_layer_size
        print(self.middle_layer_image_width)
        print(self.middle_layer_size)
        super(Model, self).__init__()
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
        self.layer5 = nn.Sequential(
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
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(
                layer_3_cnn_filters,
                layer_2_cnn_filters,
                kernel_size=layer_3_kernel_size,
                stride=[STRIDE, STRIDE, STRIDE],
                padding=int(layer_3_kernel_size / 2),
                output_padding=[1, 1, 1],
            ),
            # nn.BatchNorm2d(4),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose3d(
                layer_2_cnn_filters,
                layer_1_cnn_filters,
                kernel_size=layer_2_kernel_size,
                stride=[STRIDE, STRIDE, 1],
                padding=int(layer_2_kernel_size / 2),
                output_padding=[1, 1, 0],
            ),
            # nn.BatchNorm2d(2),
            nn.LeakyReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose3d(
                layer_1_cnn_filters,
                IMAGE_DEPTH,
                kernel_size=layer_1_kernel_size,
                stride=[1, 1, 1],
                padding=int(layer_1_kernel_size / 2),
                output_padding=[0, 0, 0],
            ),
            # nn.BatchNorm2d(4)
        )
        self.layer_force_0 = nn.Sequential(
            nn.Linear(4, middle_layer_size), nn.LeakyReLU()
        )
        self.layer_middle_0 = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_middle_1 = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_layer_size), nn.LeakyReLU()
        )
        self.layer_middle_2 = nn.Sequential(
            nn.Linear(middle_layer_size, middle_layer_size), nn.LeakyReLU()
        )
        self.layer9 = nn.Sequential(nn.Sigmoid())

    def forward(self, batch_data):
        x = batch_data["s0"]
        x_force_0 = batch_data["force_0"]
        x_force_1 = batch_data["force_1"]
        print_shape = False
        out_force_0 = self.layer_force_0(torch.cat((x_force_0, x_force_1), 1))
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
        # Add block force.
        out_combined = self.layer_middle_0(torch.add(out4_flat, out_force_0))
        out_middle_1 = self.layer_middle_1(out_combined)
        out_middle_2 = self.layer_middle_2(out_middle_1)

        out_combined_image = out_middle_2.view(
            out_middle_2.size(0),
            self.layer_4_cnn_filters,
            self.middle_layer_image_width,
            self.middle_layer_image_width,
            1,
        )
        # out3 = torch.add(self.layer3(out_combined_image), out1)
        if print_shape:
            print(out_combined_image.shape)
        out5 = self.layer5(out_combined_image)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        # logits = torch.add(self.layer4(out3), x)
        logits = self.layer8(out7)
        out_9 = self.layer9(logits)
        if print_shape:
            print(out5.shape)
            print(out6.shape)
            print(out7.shape)
            print(out_9.shape)
        return (logits, out_9)
