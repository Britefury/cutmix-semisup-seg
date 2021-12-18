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
      for augmentation driven consistency regularization.
    We separate the two modes so that when transforming pairs, geometric
    augmentation parameters can be chosen such that e.g. crops overlap,
    in order to ensure that there are image regions common to both
    crops/elements of the pair, to ensure that consistency loss can be computed.
    A paired sample is a dictionary of the form `{'sample0': <sample 0>,
    'sample1': <sample 1>}; where a single sample (or member of a
    pair) will have keys such as 'image', 'mask', 'labels', 'xf'.
    The `transform` method determines which mode to use and delegates
    to the `transform_single` or `transform_pair` methods as appropriate.
    """
    def apply(self, sample):
        if 'sample0' in sample and 'sample1' in sample:
            # Its a pair
            s0, s1 = self.transform_pair(sample['sample0'], sample['sample1'])
            return dict(sample0=s0, sample1=s1)
        else:
            return self.transform_single(sample)

    def transform_single(self, sample):
        raise NotImplementedError

    def transform_pair(self, sample0, sample1):
        return (self.transform_single(sample0), self.transform_single(sample1))


class SegTransformCompose (object):
    """
    Segmentation transform compose

    Similar to torchvision Compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def apply(self, sample):
        for t in self.transforms:
            sample = t.apply(sample)
        return sample


class SegTransformToPair (SegTransform):
    """
    Convert a single sample to a paired sample
    """
    def transform_single(self, sample):
        sample0 = sample
        sample1 = sample0.copy()
        return dict(sample0=sample0, sample1=sample1)

    def transform_pair(self, sample0, sample1):
        raise TypeError('Cannot split a paired sample into pairs again')


def get_mean_std(ds, net):
    mean, std = ds.get_mean_std()
    if net.MEAN is not None:
        mean = net.MEAN
    if net.STD is not None:
        std = net.STD
    return mean, std

