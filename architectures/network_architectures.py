import math
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils import model_zoo

try:
    from torchvision.models.segmentation import deeplabv3_resnet101
except ImportError:
    deeplabv3_resnet101 = None

from architectures import deeplab2, resunet, denseunet, deeplab3plus, mit_csail_semseg


class ArchRegistry(object):
    def __init__(self):
        self.archs = {}

    def register(self, name):
        """
        Usage:

        @registry.register('my_arch')
        def my_arch(...):
            ...
        """

        def deco(arch):
            self.archs[name] = arch
            return arch

        return deco

    def get(self, name):
        return self.archs[name]

    def names(self):
        return self.archs.keys()


seg = ArchRegistry()


@seg.register('resnet50unet_imagenet')
def resnet50unet_imagenet(num_classes, pretrained=True):
    return resunet.resnet50unet(num_classes, pretrained=pretrained)

@seg.register('resnet101unet_imagenet')
def resnet101unet_imagenet(num_classes, pretrained=True):
    return resunet.resnet101unet(num_classes, pretrained=pretrained)


@seg.register('densenet161unet')
def densenet161unet(num_classes):
    return denseunet.densenet161unet(num_classes)

@seg.register('densenet161unet_imagenet')
def densenet161unet_imagenet(num_classes):
    return denseunet.densenet161unet_imagenet(num_classes)


@seg.register('resnet101_deeplab_coco')
def resnet101_deeplab_coco(num_classes=21, pretrained=True):
    return deeplab2.resnet101_deeplab_coco(num_classes=num_classes, pretrained=pretrained)

@seg.register('resnet101_deeplab_imagenet')
def resnet101_deeplab_imagenet(num_classes=21, pretrained=True):
    return deeplab2.resnet101_deeplab_imagenet(num_classes=num_classes, pretrained=pretrained)

@seg.register('resnet101_deeplab_imagenet_mittal_std')
def resnet101_deeplab_imagenet(num_classes=21, pretrained=True):
    return deeplab2.resnet101_deeplab_imagenet_mittal_std(num_classes=num_classes, pretrained=pretrained)


@seg.register('resnet101_deeplabv3_coco')
def resnet101_deeplabv3_coco(num_classes=21, pretrained=True):
    if deeplabv3_resnet101 is None:
        raise NotImplementedError('DeepLab v3 not available on this installation; requires PyTorch 1.1 and '
                                  'torchvision 0.3')
    deeplab = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
    if pretrained:
        url_coco = 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
        state_dict = model_zoo.load_url(url_coco)
        deeplab2._load_state_into_model(deeplab, state_dict)
    return deeplab3plus.DeepLabv3Wrapper(deeplab)

@seg.register('resnet101_deeplabv3_imagenet')
def resnet101_deeplabv3_imagenet(num_classes=21, pretrained=True):
    if deeplabv3_resnet101 is None:
        raise NotImplementedError('DeepLab v3 not available on this installation; requires PyTorch 1.1 and '
                                  'torchvision 0.3')
    deeplab = deeplabv3_resnet101(pretrained=False, num_classes=num_classes)
    if pretrained:
        # We can used the DeepLab2 ResNet-101 weights URL
        state_dict = model_zoo.load_url(deeplab2._RESNET_101_IMAGENET_URL)
        state_dict = {'backbone.{}'.format(key): value for key, value in state_dict.items()}
        deeplab2._load_state_into_model(deeplab, state_dict)
    return deeplab3plus.DeepLabv3Wrapper(deeplab)


@seg.register('resnet101_deeplabv3plus_imagenet')
def resnet101_deeplabv3plus_imagenet(num_classes=21, pretrained=True):
    if deeplabv3_resnet101 is None:
        raise NotImplementedError('DeepLab v3+ not available on this installation; requires PyTorch 1.1 and '
                                  'torchvision 0.3')
    return deeplab3plus.resnet101_deeplabv3plus_imagenet(num_classes=num_classes, pretrained=pretrained)


@seg.register('resnet101_pspnet_imagenet')
def resnet101_pspnet_imagenet(num_classes=21, pretrained=True):
    model = mit_csail_semseg.MITSemSegNet(num_classes, 'resnet101dilated', 'ppm', pretraining='imagenet')
    return model


def robust_binary_crossentropy(pred, tgt, eps=1e-6):
    inv_tgt = 1.0 - tgt
    inv_pred = 1.0 - pred + eps
    return -(tgt * torch.log(pred + eps) + inv_tgt * torch.log(inv_pred))

EPS = sys.float_info.epsilon

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    # Taken from https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/ramps.py
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
