from models.blocks.conv import conv3x3
from models.head.base import BaseNet

from torch import nn


class FCN(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(FCN, self).__init__(backbone, pretrained)

        in_channels = self.backbone.out_channels
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, nclass, 1))
