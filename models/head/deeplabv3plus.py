from models.blocks.conv import conv3x3
from models.head.base import BaseNet
from models.head.deeplabv3 import ASPPModule
import torch
from torch import nn
import torch.nn.functional as F

class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(DeepLabV3Plus, self).__init__(backbone, pretrained)

        low_level_channels = self.backbone.channels[1]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, [12, 24, 36], lightweight)

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(conv3x3(high_level_channels // 8 + 48, 256, lightweight),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  #nn.Dropout(0.5, False),
                                  conv3x3(256, 256, lightweight),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        _, c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=(h // 4, w // 4), mode="bilinear", align_corners=False)

        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)
        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out