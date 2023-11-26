from models.head.deeplabv3 import DeepLabV3
from models.head.pspnet import PSPNet
from models.head.deeplabv3plus import DeepLabV3Plus
from models.head.unet import Unet2D


def get_model(model, backbone, pretrained, nclass, lightweight):
    if model == "deeplabv3":
        model = DeepLabV3(backbone, pretrained, nclass, lightweight)
    elif model == "deeplabv3plus":
        model = DeepLabV3Plus(backbone, pretrained, nclass, lightweight)
    elif model == "pspnet":
        model = PSPNet(backbone, pretrained, nclass, lightweight)
    elif model == "unet":
        model = Unet2D(num_classes=nclass)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    return model