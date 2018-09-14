from torch import nn
import torch
import torch.optim as optim
from block_sys import IMAGE_DEPTH, GRID_SIZE, FRAMES
import numpy as np

LOSS_MEAN_WINDOW = 100000
PRINT_LOSS_MEAN_ITERATION = 100

STRIDE = 2


class Trainer:
    def __init__(self, learning_rate, model):
        self.criterion = nn.BCEWithLogitsLoss()
        self.iteration = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.running_loss = np.ones(LOSS_MEAN_WINDOW)
        self.running_loss_idx = 0
        print(self.model)

    def train(self, x, y, x_force_0, x_force_1):
        self.optimizer.zero_grad()
        logits, out = self.model.forward(x, x_force_0, x_force_1)
        loss = self.criterion(
            logits.reshape([logits.size(0), -1]), y.reshape([y.size(0), -1])
        )
        loss.backward()
        self.running_loss[self.running_loss_idx] = loss.data[0]
        if self.running_loss_idx >= LOSS_MEAN_WINDOW - 1:
            self.running_loss_idx = 0
        else:
            self.running_loss_idx += 1
        mean_loss = np.sum(self.running_loss) / LOSS_MEAN_WINDOW
        if (self.iteration % PRINT_LOSS_MEAN_ITERATION) == 0:
            print("loss: {}".format(mean_loss))
        self.optimizer.step()
        self.iteration += 1
        return (out.data, mean_loss)


class Model0(nn.Module):
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
        super(Model0, self).__init__()
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
            nn.Linear(4, force_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_force_1 = nn.Sequential(
            nn.Linear(force_hidden_layer_size, int(middle_hidden_layer_size / 2)),
            nn.LeakyReLU(),
        )
        self.layer_middle_0 = nn.Sequential(
            nn.Linear(middle_layer_size, int(middle_hidden_layer_size / 2)),
            nn.LeakyReLU(),
        )
        self.layer_middle_1 = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_layer_size), nn.LeakyReLU()
        )
        self.layer9 = nn.Sequential(nn.Sigmoid())

    def forward(self, x, x_force_0, x_force_1):
        print_shape = False
        out_force_0 = self.layer_force_0(torch.cat((x_force_0, x_force_1), 1))
        out_force_1 = self.layer_force_1(out_force_0)
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
        out_middle_0 = self.layer_middle_0(out4_flat)
        # Concatonate block force.
        # out_combined = torch.add(torch.add(out2_flat, out_force_0), out_force_1)
        out_combined = self.layer_middle_1(torch.cat((out_middle_0, out_force_1), 1))

        # out_combined = torch.add(out2_flat, out_force_1)
        out_combined_image = out_combined.view(
            out_combined.size(0),
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


class Trainer1:
    def __init__(self, learning_rate, model):
        self.criterion = nn.MSELoss()
        self.iteration = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.running_loss = np.ones(LOSS_MEAN_WINDOW)
        self.running_loss_idx = 0
        print(self.model)

    def train(self, x, y, x_force_0, x_force_1):
        self.optimizer.zero_grad()
        out = self.model.forward(x, x_force_0, x_force_1)
        loss = self.criterion(out, y)
        loss.backward()
        self.running_loss[self.running_loss_idx] = loss.data[0]
        if self.running_loss_idx >= LOSS_MEAN_WINDOW - 1:
            self.running_loss_idx = 0
        else:
            self.running_loss_idx += 1
        mean_loss = np.sum(self.running_loss) / LOSS_MEAN_WINDOW
        if (self.iteration % PRINT_LOSS_MEAN_ITERATION) == 0:
            print("loss: {}".format(mean_loss))
        self.optimizer.step()
        del loss
        self.iteration += 1
        return (out.data, mean_loss)


class Model1(nn.Module):
    def __init__(self):
        """
        force_add determines whether the force is added or concatonated.
        """
        super(Model1, self).__init__()
        self.layer_0 = nn.Sequential(nn.Linear(8, 16), nn.LeakyReLU())
        self.layer_1 = nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU())
        self.layer_3 = nn.Sequential(nn.Linear(16, 8))
        self.layer_force_0_0 = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU())
        self.layer_force_0_1 = nn.Sequential(nn.Linear(8, 8), nn.LeakyReLU())
        self.layer_force_1_0 = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU())
        self.layer_force_1_1 = nn.Sequential(nn.Linear(8, 8), nn.LeakyReLU())

    def forward(self, x, x_force_0, x_force_1):
        out_force_0_0 = self.layer_force_0_0(x_force_0)
        out_force_1_0 = self.layer_force_1_0(x_force_1)
        out_force_0_1 = self.layer_force_0_1(out_force_0_0)
        out_force_1_1 = self.layer_force_1_1(out_force_1_0)
        out1 = self.layer_0(x)
        out2 = self.layer_1(out1)
        # Concatonate block force.
        # out_combined = torch.add(torch.add(out2, out_force_0_1), out_force_1_1)
        out_combined = torch.cat((out2, out_force_0_1, out_force_1_1), 1)

        # out_combined = torch.add(out2_flat, out_force_1)
        # out_combined_image = out_combined.view(out_combined.size(0), 32, 8, 8)
        # print(out_combined_image.shape)
        # out3 = torch.add(self.layer3(out_combined_image), out1)
        out3 = self.layer_2(out_combined)
        # logits = torch.add(self.layer4(out3), x)
        out4 = self.layer_3(out3)
        return out4
