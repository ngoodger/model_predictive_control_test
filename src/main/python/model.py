import torch
import trainer
from block_sys import GRID_SIZE, IMAGE_DEPTH
from torch import nn

STRIDE = 2


class ModelTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, model):
        super(ModelTrainer, self).__init__(learning_rate, model)

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_loss(self, batch_data):
        for i in range(batch_data["seq_len"] - 1):
            s_initial = batch_data["s"][0]
            force_0 = batch_data["force"][i]
            force_1 = batch_data["force"][i + 1]
            if i == 0:
                logits, out, recurrent_state = self.nn_module.forward(
                    s_initial, None, force_0, force_1, first_run=True
                )
            else:
                logits, out, recurrent_state = self.nn_module.forward(
                    None, recurrent_state, force_0, force_1, first_run=False
                )
            y = batch_data["s"][i + 1]
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
        recurrent_layer_size,
    ):
        """
        force_add determines whether the force is added or concatonated.
        """

        super(Model, self).__init__()
        self.layer_1_cnn_filters = layer_1_cnn_filters
        self.layer_2_cnn_filters = layer_2_cnn_filters
        self.layer_3_cnn_filters = layer_3_cnn_filters
        self.layer_4_cnn_filters = layer_4_cnn_filters
        self.layer_1_kernel_size = layer_1_kernel_size
        self.layer_2_kernel_size = layer_2_kernel_size
        self.layer_3_kernel_size = layer_3_kernel_size
        self.layer_4_kernel_size = layer_4_kernel_size
        self.middle_hidden_layer_size = middle_hidden_layer_size
        self.recurrent_layer_size = recurrent_layer_size
        self.init_rnn = torch.nn.Parameter(
            torch.rand(middle_hidden_layer_size), requires_grad=True
        )
        LAYERS = 4
        self.middle_layer_image_width = int(GRID_SIZE / (2 ** (LAYERS - 1)))
        middle_layer_size = int(
            layer_4_cnn_filters * 1 * (self.middle_layer_image_width ** 2)
        )
        self.middle_layer_size = middle_layer_size
        self.cnn_layer_0 = nn.Sequential(
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
        self.cnn_layer_1 = nn.Sequential(
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
        self.cnn_layer_2 = nn.Sequential(
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
        self.cnn_layer_3 = nn.Sequential(
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
        self.tcnn_layer_0 = nn.Sequential(
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
        self.tcnn_layer_1 = nn.Sequential(
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
        self.tcnn_layer_2 = nn.Sequential(
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
        self.tcnn_layer_3 = nn.Sequential(
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
        self.layer_force_recurrent = nn.Sequential(
            nn.Linear(4, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_cnn_recurrent = nn.Sequential(
            nn.Linear(middle_layer_size, middle_hidden_layer_size), nn.LeakyReLU()
        )
        self.layer_recurrent_hidden = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_recurrent_out = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_layer_size), nn.LeakyReLU()
        )
        self.layer_recurrent_hidden = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_recurrent = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_hidden_layer_size),
            nn.LeakyReLU(),
        )
        self.layer_sigmoid_out = nn.Sequential(nn.Sigmoid())

    def forward(
        self, s_initial, last_recurrent_state, force_0, force_1, first_run=False
    ):
        if first_run:
            x = s_initial
        else:
            x = None
        x_force_0 = force_0
        x_force_1 = force_1
        if first_run:
            recurrent_state = self.init_rnn
        else:
            recurrent_state = last_recurrent_state
        out_force_0 = self.layer_force_recurrent(torch.cat((x_force_0, x_force_1), 1))
        recurrent_out = self.layer_recurrent(recurrent_state)
        if first_run:
            out1 = self.cnn_layer_0(x)
            out2 = self.cnn_layer_1(out1)
            out3 = self.cnn_layer_2(out2)
            out4 = self.cnn_layer_3(out3)
            out4_flat = out4.view(out4.size(0), -1)
            out_combined = self.layer_cnn_recurrent(out4_flat)
            recurrent_hidden = self.layer_recurrent_hidden(
                torch.add(torch.add(out_combined, recurrent_out), out_force_0)
            )
        else:
            recurrent_hidden = self.layer_recurrent_hidden(
                torch.add(recurrent_out, out_force_0)
            )
        recurrent_out = self.layer_recurrent(recurrent_hidden)
        out_flat = self.layer_recurrent_out(recurrent_out)

        out_image = out_flat.view(
            out_flat.size(0),
            self.layer_4_cnn_filters,
            self.middle_layer_image_width,
            self.middle_layer_image_width,
            1,
        )
        out5 = self.tcnn_layer_0(out_image)
        out6 = self.tcnn_layer_1(out5)
        out7 = self.tcnn_layer_2(out6)
        logits = self.tcnn_layer_3(out7)
        out_9 = self.layer_sigmoid_out(logits)
        return (logits, out_9, recurrent_out)
