from models.blocks.conv import conv3x3
from models.head.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(DeepLabV3, self).__init__(backbone, pretrained)

        self.head = DeepLabV3Head(self.backbone.out_channels, nclass, lightweight)


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight, atrous_rates=[6, 12, 18]):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPPModule(in_channels, atrous_rates, lightweight)

        self.block = nn.Sequential(conv3x3(inter_channels, inter_channels, lightweight),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


def ASPPConv(in_channels, out_channels, atrous_rate, lightweight):
    block = nn.Sequential(conv3x3(in_channels, out_channels, lightweight, atrous_rate),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=False)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, lightweight):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, lightweight)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, lightweight)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, lightweight)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
