import torch
from torch import nn


class VariationalLastLayer(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.mean = nn.Conv2d(in_channels, out_features, kernel_size=1, padding=0)
        self.logvar = nn.Conv2d(in_channels, out_features, kernel_size=1, padding=0)

    def forward(self, x):
        mean = self.mean(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar
