# Decoder based off the decoder used in Dense U-net:
# https://github.com/xmengli999/H-DenseUNet/blob/master/denseunet.py

import numpy as np
import torch.nn as nn, torch.nn.functional as F
from torchvision import models
from architectures.util import freeze_bn_module


class DecoderBlock(nn.Module):
    def __init__(self, x_chn_in, skip_chn_in, chn_out):
        super(DecoderBlock, self).__init__()

        if x_chn_in != skip_chn_in:
            raise ValueError('x_chn_in != skip_chn_in')

        self.x_chn_in = x_chn_in
        self.skip_chn_in = skip_chn_in
        self.chn_out = chn_out

        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(x_chn_in, chn_out, 3, padding=1, bias=False)
        self.conv_bn = nn.BatchNorm2d(chn_out)

    def forward(self, x_in, skip_in):
        if x_in.shape[1] != self.x_chn_in:
            raise ValueError('x_in.shape[1]={}, self.x_chn_in={}'.format(x_in.shape[1], self.x_chn_in))
        if skip_in.shape[1] != self.skip_chn_in:
            raise ValueError('skip_in.shape[1]={}, self.skip_chn_in={}'.format(skip_in.shape[1], self.skip_chn_in))
        x_up = self.up(x_in)
        x = x_up + skip_in
        x = F.relu(self.conv_bn(self.conv(x)))
        return x


class ResUNet(nn.Module):
    BLOCK_SIZE = (32, 32)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, base_model, num_classes, pretrained):
        super(ResUNet, self).__init__()

        # Assign base model
        self.base_model = base_model

        self.pretrained = pretrained

        # A 1x1 conv layer mapping # features from denseblock3 to # of features from norm5
        self.line0_conv = nn.Conv2d(2048, 1024, 1)

        # self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(1024, 1024, 512)
        self.decoder2 = DecoderBlock(512, 512, 256)
        self.decoder1 = DecoderBlock(256, 256, 64)
        self.decoder0 = DecoderBlock(64, 64, 64)

        # Final part: upsample x2, a single 64 channel 3x3 conv layer, dropout, BN and finally pixel classification
        self.final_dec_up = nn.Upsample(scale_factor=2)
        self.final_dec_conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.final_dec_drop = nn.Dropout(0.3)
        self.final_dec_bn = nn.BatchNorm2d(64)

        # Final pixel classification layer
        self.final_clf = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.base_model.conv1(x)
        r2 = x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        r4 = x = self.base_model.layer1(x)
        r8 = x = self.base_model.layer2(x)
        r16 = x = self.base_model.layer3(x)
        r32 = x = self.base_model.layer4(x)

        # Apply line0_conv from last tap
        x = self.line0_conv(x)

        # Apply decoder blocks
        x = self.decoder3(x, r16)
        x = self.decoder2(x, r8)
        x = self.decoder1(x, r4)
        x = self.decoder0(x, r2)

        # Final layers
        x = self.final_dec_bn(self.final_dec_drop(self.final_dec_conv(self.final_dec_up(x))))
        x = F.relu(x)
        logits = self.final_clf(x)

        return logits

    def pretrained_parameters(self):
        if self.pretrained:
            return list(self.base_model.parameters())
        else:
            return []

    def new_parameters(self):
        if self.pretrained:
            pretrained_ids = [id(p) for p in self.base_model.parameters()]
            return [p for p in self.parameters() if id(p) not in pretrained_ids]
        else:
            return list(self.parameters())

    def freeze_batchnorm(self):
        self.base_model.apply(freeze_bn_module)


def resnet50unet(num_classes, pretrained=True):
    base_model = models.resnet50(pretrained=pretrained)
    return ResUNet(base_model, num_classes, pretrained=pretrained)

def resnet101unet(num_classes, pretrained=True):
    base_model = models.resnet101(pretrained=pretrained)
    return ResUNet(base_model, num_classes, pretrained=pretrained)
