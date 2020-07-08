# Used as reference:
# https://github.com/xmengli999/H-DenseUNet/blob/master/denseunet.py
# Uses torchvision densenet as the encoder

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
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


class DenseUNet(nn.Module):
    BLOCK_SIZE = (32, 32)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, base_model, num_classes, mean, std, pretrained):
        super(DenseUNet, self).__init__()

        self.MEAN = mean
        self.STD = std
        self.pretrained = pretrained

        # Module names for taps within encoder
        self.tap_names = []
        # Per tap number of channels
        enc_chn = []

        # Assign base model
        self.base_model = base_model

        # Tap after features.norm0 (next layer is pool0)
        enc_chn.append(base_model.features.norm0.num_features)
        self.tap_names.append('pool0')

        # Tap after features.denseblock1 (next layer is transition1)
        enc_chn.append(base_model.features.transition1.norm.num_features)
        self.tap_names.append('transition1')

        # Tap after features.denseblock2 (next layer is transition2)
        enc_chn.append(base_model.features.transition2.norm.num_features)
        self.tap_names.append('transition2')

        # Tap after features.denseblock3 (next layer is transition3)
        enc_chn.append(base_model.features.transition3.norm.num_features)
        self.tap_names.append('transition3')

        # Number of channels out of features model
        n_chn = base_model.features.norm5.num_features

        # A 1x1 conv layer mapping # features from denseblock3 to # of features from norm5
        self.line0_conv = nn.Conv2d(enc_chn[-1], n_chn, 1)
        enc_chn[-1] = n_chn

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        # We build the decoder in the reverse order in comparison to that of the encoder,
        # so reverse the order of `enc_chn`
        enc_chn = enc_chn[::-1]

        # Build the decoder blocks
        for e_chn_a, e_chn_b in zip(enc_chn, enc_chn[1:] + enc_chn[-1:]):
            decoder = DecoderBlock(n_chn, e_chn_a, e_chn_b)
            self.decoder_blocks.append(decoder)
            n_chn = e_chn_b

        # Reverse the order of the decoder block
        self.decoder_blocks = self.decoder_blocks[::-1]

        # Final part: upsample x2, a single 64 channel 3x3 conv layer, dropout, BN and finally pixel classification
        self.final_dec_up = nn.Upsample(scale_factor=2)
        self.final_dec_conv = nn.Conv2d(n_chn, 64, 3, padding=1, bias=False)
        self.final_dec_drop = nn.Dropout(0.3)
        self.final_dec_bn = nn.BatchNorm2d(64)

        # Final pixel classification layer
        self.final_clf = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Apply layers from base_model.features, taking tensors at tap points
        enc_x = []
        for name, mod in self.base_model.features.named_children():
            if name in self.tap_names:
                enc_x.append(x)
            x = mod(x)
        # ReLu after encoder BN layer
        x = F.relu(x)

        # Apply line0_conv from last tap
        line0 = self.line0_conv(enc_x[-1])
        enc_x[-1] = line0

        # Apply decoder blocks in reverse order
        for dec_block, ex in zip(self.decoder_blocks[::-1], enc_x[::-1]):
            x = dec_block(x, ex)

        # Final layers
        x = self.final_dec_bn(self.final_dec_drop(self.final_dec_conv(self.final_dec_up(x))))
        x = F.relu(x)
        logits = self.final_clf(x)

        return logits

    def pretrained_parameters(self):
        if self.pretrained:
            return list(self.base_model.features.parameters())
        else:
            return []

    def new_parameters(self):
        if self.pretrained:
            pretrained_ids = [id(p) for p in self.base_model.features.parameters()]
            return [p for p in self.parameters() if id(p) not in pretrained_ids]
        else:
            return list(self.parameters())

    def freeze_batchnorm(self):
        self.base_model.apply(freeze_bn_module)


def densenet161unet(num_classes):
    base_model = models.densenet161(pretrained=False)
    return DenseUNet(base_model, num_classes, mean=None, std=None, pretrained=False)

def densenet161unet_imagenet(num_classes):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    base_model = models.densenet161(pretrained=True)
    return DenseUNet(base_model, num_classes, mean=mean, std=std, pretrained=True)

