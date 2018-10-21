import torch
import trainer
from block_sys import GRID_SIZE, IMAGE_DEPTH
from torch import nn
import inspect
import hashlib

STRIDE = 2
LSTM_DEPTH = 2


def hash_model():
    print(__name__)
    module_src_code = inspect.getsource(__import__(__name__)).encode("utf-8")
    print(module_src_code)
    return hashlib.sha256(module_src_code).hexdigest()


# Can not easily add as model function because it is not supported by data parallel.
def forward_sequence(model, batch_data, use_label_output=False):
    logits_list = []
    out_list = []
    observation = batch_data["observations"][0]
    for i in range(batch_data["seq_len"] - 1):
        force_0 = batch_data["forces"][i]
        force_1 = batch_data["forces"][i + 1]
        if i == 0:
            logits, out, recurrent_state = model.forward(
                observation, None, force_0, force_1, first_iteration=True
            )
        else:
            logits, out, recurrent_state = model.forward(
                observation, recurrent_state, force_0, force_1, first_iteration=False
            )
        if use_label_output:
            observation = batch_data["observations"][i + 1]
        else:
            observation = out
        out_list.append(out)
        logits_list.append(logits)
    return logits_list, out_list


class ModelTrainer(trainer.BaseTrainer):
    def __init__(self, learning_rate, model):
        super(ModelTrainer, self).__init__(learning_rate, model)

    def get_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def get_loss(self, batch_data):
        loss = 0.
        logits_list, _ = forward_sequence(
            self.nn_module, batch_data, use_label_output=True
        )
        loss = sum(
            self.criterion(
                logits_list[i].reshape([logits_list[i].size(0), -1]),
                batch_data["observations"][i + 1].reshape(
                    [batch_data["observations"][i + 1].size(0), -1]
                ),
            )
            for i in range(batch_data["seq_len"] - 1)
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
        self.init_recurrent_state = (
            torch.nn.Parameter(
                torch.rand(LSTM_DEPTH, 1, middle_hidden_layer_size), requires_grad=True
            ),
            torch.nn.Parameter(
                torch.rand(LSTM_DEPTH, 1, middle_hidden_layer_size), requires_grad=True
            ),
        )
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
        self.layer_tcnn_1 = nn.Sequential(
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
        self.layer_tcnn_2 = nn.Sequential(
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
        self.layer_tcnn_3 = nn.Sequential(
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
        self.layer_recurrent_out = nn.Sequential(
            nn.Linear(middle_hidden_layer_size, middle_layer_size), nn.LeakyReLU()
        )
        self.layer_recurrent = nn.LSTM(
            middle_hidden_layer_size, middle_hidden_layer_size, LSTM_DEPTH
        )
        self.layer_sigmoid_out = nn.Sequential(nn.Sigmoid())

    def forward(
        self, observation, last_recurrent_state, force_0, force_1, first_iteration=False
    ):

        out_force_recurrent = self.layer_force_recurrent(
            torch.cat((force_0, force_1), 1)
        )

        # On the first iteration of the RNN
        # use the learned self.init_recurrent_tate parameter as the initial recurrent state.
        # Only feed through the initial frame on the first iteration since the model must
        # rely on latent state to predict future outputs..
        x = observation
        out_cnn_0 = self.layer_cnn_0(x)
        out_cnn_1 = self.layer_cnn_1(out_cnn_0)
        out_cnn_2 = self.layer_cnn_2(out_cnn_1)
        out_cnn_3 = self.layer_cnn_3(out_cnn_2)
        out_input_image_flat = out_cnn_3.view(out_cnn_3.size(0), -1)
        out_cnn_recurrent = self.layer_cnn_recurrent(out_input_image_flat)
        if first_iteration:
            last_recurrent_state = self.init_recurrent_state

        # Combine outputs from previous recurrent state and force layer.
        combined = torch.add(out_cnn_recurrent, out_force_recurrent)
        out_recurrent, out_recurrent_state = self.layer_recurrent(
            combined.view(1, 1, -1), last_recurrent_state
        )
        out_image_flat_hidden = self.layer_recurrent_out(out_recurrent)

        out_image_hidden = out_image_flat_hidden.view(
            out_image_flat_hidden.size(0),
            self.layer_4_cnn_filters,
            self.middle_layer_image_width,
            self.middle_layer_image_width,
            1,
        )
        out_tcnn_0 = self.layer_tcnn_0(out_image_hidden)
        out_tcnn_1 = self.layer_tcnn_1(out_tcnn_0)
        out_tcnn_2 = self.layer_tcnn_2(out_tcnn_1)
        out_logits = self.layer_tcnn_3(out_tcnn_2)
        out_sigmoid = self.layer_sigmoid_out(out_logits)
        return (out_logits, out_sigmoid, out_recurrent_state)
