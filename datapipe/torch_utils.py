from distutils.version import LooseVersion, StrictVersion
import torch

# align_corners option available for v1.3.0 and above
HAS_AFFINE_ALIGN_CORNERS = LooseVersion(torch.__version__) >= LooseVersion('1.3.0')
# align_corners defaults to True before v1.4.0, False from v1.4.0 and after
AFFINE_ALIGN_CORNERS_DEFAULT = LooseVersion(torch.__version__) <= LooseVersion('1.3.0')


def affine_align_corners_kw(val):
    if HAS_AFFINE_ALIGN_CORNERS:
        return dict(align_corners=val)
    else:
        if not val:
            raise RuntimeError('align_corners not available in torch version {} so '
                               'cannot set to False'.format(torch.__version__))
        return {}
