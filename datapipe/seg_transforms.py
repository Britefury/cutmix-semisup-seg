"""
Segmentation data pipeline transform base classes
"""


class SegTransform (object):
    """
    Segmentation transform base class

    Designed along the same lines as torchvision transforms,
    except there are two modes:
    - transform a single image
    - transform a paired image with different augmentation parameters
      for augmentation driven consistency regularization
    A method is provided for each mode.
    """
    def transform_single(self, sample):
        raise NotImplementedError

    def transform_pair(self, sample0, sample1):
        return (self.transform_single(sample0), self.transform_single(sample1))


class SegTransformCompose (SegTransform):
    """
    Segmentation transform compose

    Similar to torchvision Compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def transform_single(self, seg_img0):
        for t in self.transforms:
            seg_img0 = t.transform_single(seg_img0)
        return seg_img0

    def transform_pair(self, sample0, sample1):
        for t in self.transforms:
            sample0, sample1 = t.transform_pair(sample0, sample1)
        return sample0, sample1


def get_mean_std(ds, net):
    mean, std = ds.get_mean_std()
    if net.MEAN is not None:
        mean = net.MEAN
    if net.STD is not None:
        std = net.STD
    return mean, std

