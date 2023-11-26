import torch
from torch import nn


class NonLocal(nn.Module):
    def __init__(self, in_channels, ratio=2):
        super(NonLocal, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // ratio

        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, 1)
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, 1)

        self.recover = nn.Sequential(nn.Conv2d(self.inter_channels, self.in_channels, 1),
                                     nn.BatchNorm2d(self.in_channels))
        nn.init.constant_(self.recover[1].weight, 0)
        nn.init.constant_(self.recover[1].bias, 0)

    def forward(self, x):
        b, _, h, w = x.shape
        c = self.inter_channels

        theta_x = self.theta(x).reshape(b, c, -1).permute(0, 2, 1)
        phi_x = self.phi(x).reshape(b, c, -1)
        g_x = self.g(x).reshape(b, c, -1).permute(0, 2, 1)

        attention = torch.matmul(theta_x, phi_x)
        attention = attention / attention.shape[-1]

        out = torch.matmul(attention, g_x).permute(0, 2, 1).reshape(b, c, h, w)
        out = self.recover(out)
        out += x

        return out