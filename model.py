from torch import nn
import torch
import torch.optim as optim
import numpy as np

LOSS_MEAN_WINDOW = 10000
PRINT_LOSS_MEAN_ITERATION = 100


class Trainer():
    def __init__(self, learning_rate, model):
        self.criterion = nn.BCEWithLogitsLoss()
        self.iteration = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate)
        self.running_loss = np.ones(LOSS_MEAN_WINDOW)
        self.running_loss_idx = 0
        print(self.model)

    def train(self, x, y, x_force_0, x_force_1):
        self.optimizer.zero_grad()
        logits, out = self.model.forward(x, x_force_0, x_force_1)
        loss = self.criterion(logits.reshape([logits.size(0), 32 * 128]),
                              y.reshape([y.size(0), 32 * 128]))
        loss.backward()
        self.running_loss[self.running_loss_idx] = loss.data[0]
        if self.running_loss_idx >= LOSS_MEAN_WINDOW - 1:
            self.running_loss_idx = 0
        else:
            self.running_loss_idx += 1
        mean_loss = np.sum(self.running_loss) / LOSS_MEAN_WINDOW
        if (self.iteration % PRINT_LOSS_MEAN_ITERATION) == 0:
            print('loss: {}'.format(mean_loss))
        self.optimizer.step()
        del loss
        self.iteration += 1
        return (out.data, mean_loss)


class Model0(nn.Module):
    def __init__(self):
        """
        force_add determines whether the force is added or concatonated.
        """
        super(Model0, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=6, stride=2, padding=2),
            # nn.BatchNorm2d(4),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=6, stride=2, padding=2,
                               output_padding=0),
            # nn.BatchNorm2d(2),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            # nn.BatchNorm2d(4)
            )
        self.layer_force_0 = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU())
        self.layer_force_1 = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU())
        self.layer_middle_0 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU())
        self.layer_middle_1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Sigmoid())

    def forward(self, x, x_force_0, x_force_1):
        out_force_0 = self.layer_force_0(torch.cat((x_force_0, x_force_1), 1))
        out_force_1 = self.layer_force_1(out_force_0)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2_flat = out2.view(out2.size(0), -1)
        out_middle_0 = self.layer_middle_0(out2_flat)
        # Concatonate block force.
        # out_combined = torch.add(torch.add(out2_flat, out_force_0), out_force_1)
        out_combined = self.layer_middle_1(torch.cat((out_middle_0, out_force_1), 1))

        # out_combined = torch.add(out2_flat, out_force_1)
        out_combined_image = out_combined.view(out_combined.size(0), 8, 8, 8)
        # print(out_combined_image.shape)
        # out3 = torch.add(self.layer3(out_combined_image), out1)
        out3 = self.layer3(out_combined_image)
        # logits = torch.add(self.layer4(out3), x)
        logits = self.layer4(out3)
        out_5 = self.layer5(logits)
        return (logits, out_5)

class Trainer1():
    def __init__(self, learning_rate, model):
        self.criterion = nn.MSELoss()
        self.iteration = 0
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate)
        self.running_loss = np.ones(LOSS_MEAN_WINDOW)
        self.running_loss_idx = 0
        print(self.model)

    def train(self, x, y, x_force_0, x_force_1):
        self.optimizer.zero_grad()
        out = self.model.forward(x, x_force_0, x_force_1)
        loss = self.criterion(out,
                              y)
        loss.backward()
        self.running_loss[self.running_loss_idx] = loss.data[0]
        if self.running_loss_idx >= LOSS_MEAN_WINDOW - 1:
            self.running_loss_idx = 0
        else:
            self.running_loss_idx += 1
        mean_loss = np.sum(self.running_loss) / LOSS_MEAN_WINDOW
        if (self.iteration % PRINT_LOSS_MEAN_ITERATION) == 0:
            print('loss: {}'.format(mean_loss))
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
        self.layer_0 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU())
        self.layer_1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU())
        self.layer_2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU())
        self.layer_3 = nn.Sequential(
            nn.Linear(16, 8))
        self.layer_force_0_0 = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU())
        self.layer_force_0_1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU())
        self.layer_force_1_0 = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU())
        self.layer_force_1_1 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU())

    def forward(self, x, x_force_0, x_force_1):
        out_force_0_0 = self.layer_force_0_0(x_force_0)
        out_force_1_0 = self.layer_force_1_0(x_force_1)
        out_force_0_1 = self.layer_force_0_1(out_force_0_0)
        out_force_1_1 = self.layer_force_1_1(out_force_1_0)
        out1 = self.layer_0(x)
        out2 = self.layer_1(out1)
        # Concatonate block force.
        #out_combined = torch.add(torch.add(out2, out_force_0_1), out_force_1_1)
        out_combined = torch.cat((out2, out_force_0_1, out_force_1_1), 1)

        # out_combined = torch.add(out2_flat, out_force_1)
        #out_combined_image = out_combined.view(out_combined.size(0), 32, 8, 8)
        # print(out_combined_image.shape)
        # out3 = torch.add(self.layer3(out_combined_image), out1)
        out3 = self.layer_2(out_combined)
        # logits = torch.add(self.layer4(out3), x)
        out4 = self.layer_3(out3)
        return out4
