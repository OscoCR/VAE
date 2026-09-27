import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels * 4, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)
