from torch import nn
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer3 = nn.Linear(16*16*16, 32)
        self.layer4 = nn.Linear(32, 16*16*16)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2,
                               output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2,
                               output_padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU())

    def forward(self, x_np):
        x = torch.from_numpy(x_np)
        print(x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(1, 16, 8, 32)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
