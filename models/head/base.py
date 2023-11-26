import torch
from models.backbone.mobilenetv2 import mobilenet_v2
from models.backbone.resnet import resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d

from torch import nn
import torch.nn.functional as F


def get_backbone(backbone, pretrained):
    if backbone == "mobilenetv2":
        backbone = mobilenet_v2(pretrained)
    elif backbone == "resnet34":
        backbone = resnet34(pretrained)
    elif backbone == "resnet50":
        backbone = resnet50(pretrained)
    elif backbone == "resnet101":
        backbone = resnet101(pretrained)
    elif backbone == "resnet152":
        backbone = resnet152(pretrained)
    elif backbone == "resnext50":
        backbone = resnext50_32x4d(pretrained)
    elif backbone == "resnext101":
        backbone = resnext101_32x8d(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone


class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained)
    
    def base_forward(self, x):
        h, w = x.shape[-2:]

        x = self.backbone.base_forward(x)[-1]
        x = self.head(x)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)

        return x

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            out = F.softmax(self.base_forward(x), dim=1)
            origin_x = x.clone()

            x = origin_x.flip(2)
            out += F.softmax(self.base_forward(x), dim=1).flip(2)

            x = origin_x.flip(3)
            out += F.softmax(self.base_forward(x), dim=1).flip(3)

            x = origin_x.transpose(2, 3).flip(3)
            out += F.softmax(self.base_forward(x), dim=1).flip(3).transpose(2, 3)

            x = origin_x.flip(3).transpose(2, 3)
            out += F.softmax(self.base_forward(x), dim=1).transpose(2, 3).flip(3)

            x = origin_x.flip(2).flip(3)
            out += F.softmax(self.base_forward(x), dim=1).flip(3).flip(2)

            out /= 6.0

            return out
