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

    def train(self, x, y, x_force):
        self.optimizer.zero_grad()
        logits, out = self.model.forward(x, x_force)
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
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2, padding=2,
                               output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(4)
            )
        self.layer_force = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Sigmoid())

    def forward(self, x, x_force):
        out_force = self.layer_force(x_force)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2_flat = out2.view(out2.size(0), -1)
        # Concatonate block force.
        out_combined = torch.add(out2_flat, out_force)
        out_combined_image = out_combined.view(out_combined.size(0), 16, 8, 8)
        out3 = torch.add(self.layer3(out_combined_image), out1)
        logits = torch.add(self.layer4(out3), x)
        out_5 = self.layer5(logits)
        return (logits, out_5)
