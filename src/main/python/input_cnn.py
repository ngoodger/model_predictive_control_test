from torch import nn
from block_sys import IMAGE_DEPTH

STRIDE = 2


class InputCNN(nn.Module):
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
    ):
        """
        force_add determines whether the force is added or concatonated.
        """

        super(InputCNN, self).__init__()
        self.layer_1_cnn_filters = layer_1_cnn_filters
        self.layer_2_cnn_filters = layer_2_cnn_filters
        self.layer_3_cnn_filters = layer_3_cnn_filters
        self.layer_4_cnn_filters = layer_4_cnn_filters
        self.layer_1_kernel_size = layer_1_kernel_size
        self.layer_2_kernel_size = layer_2_kernel_size
        self.layer_3_kernel_size = layer_3_kernel_size
        self.layer_4_kernel_size = layer_4_kernel_size

        self.layer_cnn_0 = nn.Sequential(
            nn.Conv3d(
                IMAGE_DEPTH,
                layer_1_cnn_filters,
                kernel_size=layer_1_kernel_size,
                stride=[1, 1, 1],
                padding=int(layer_1_kernel_size / 2),
            ),
            # nn.BatchNorm2d(2),
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
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, observation):
        out_cnn_0_act = self.layer_cnn_0(observation)
        out_cnn_0 = self.leaky_relu(out_cnn_0_act)
        out_cnn_1_act = self.layer_cnn_1(out_cnn_0)
        out_cnn_1 = self.leaky_relu(out_cnn_1_act)
        out_cnn_2_act = self.layer_cnn_2(out_cnn_1)
        out_cnn_2 = self.leaky_relu(out_cnn_2_act)
        out_cnn_3_act = self.layer_cnn_3(out_cnn_2)
        out_cnn_3 = self.leaky_relu(out_cnn_3_act)
        out_input_cnn_flat = out_cnn_3.view(out_cnn_3.size(0), -1)
        return out_input_cnn_flat
