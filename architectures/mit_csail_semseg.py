# Wrappers for MIT CSAIL Semantic Segmentation library for PyTorch
# At https://github.com/CSAILVision/semantic-segmentation-pytorch/pull/239
# Here we use a modified version:
# https://github.com/Britefury/semantic-segmentation-pytorch
# Specifically the `logits-from-models` branch

import numpy as np
import torch.nn as nn
from architectures.deeplab2 import freeze_bn_module

try:
    import mit_semseg.models as mss_models
except ImportError:
    mss_models = None

class MITSemSegNet (nn.Module):
    BLOCK_SIZE = (1, 1)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, num_classes, encoder_arch, decoder_arch, pretraining=None):
        super(MITSemSegNet, self).__init__()

        if mss_models is None:
            raise NotImplementedError('MIT CSAIL Semantic Segmentation library not available on this installation')

        if encoder_arch == 'mobilenetv2dilated':
            fc_dim = 320
        elif encoder_arch in {'resnet18', 'resnet18dilated'}:
            fc_dim = 512
        elif encoder_arch == 'hrnetv2':
            fc_dim = 720
        elif encoder_arch in {'resnet50', 'resnet50dilated', 'resnet101', 'resnet101dilated', 'resnext101'}:
            fc_dim = 2048
        else:
            raise ValueError('Unknown encoder architecture {}; cannot determine fc_dim'.format(encoder_arch))


        self.encoder = mss_models.ModelBuilder.build_encoder(
            arch=encoder_arch, fc_dim=fc_dim, weights='')
        self.decoder = mss_models.ModelBuilder.build_decoder(
            arch=decoder_arch, fc_dim=fc_dim, num_class=num_classes, weights='')

        self.pretraining = pretraining


    def forward(self, x, feature_maps=False, use_dropout=False):
        seg_size = x.shape[-2:]
        pred_dict = self.decoder(self.encoder(x, return_feature_maps=True), segSize=seg_size)
        return pred_dict['logits']


    def freeze_batchnorm(self):
        self.encoder.apply(freeze_bn_module)


    def pretrained_parameters(self):
        if self.pretraining is None:
            return []
        elif self.pretraining == 'imagenet':
            return list(self.encoder.parameters())
        else:
            raise ValueError('Unknown pretraining {}'.format(self.pretraining))

    def new_parameters(self):
        if self.pretraining is None:
            return list(self.parameters())
        elif self.pretraining == 'imagenet':
            return list(self.decoder.parameters())
        else:
            raise ValueError('Unknown pretraining {}'.format(self.pretraining))

