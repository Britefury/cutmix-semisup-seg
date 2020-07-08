import numpy as np

def identity_xf(N):
    """
    Construct N identity 2x3 transformation matrices
    :return: array of shape (N, 2, 3)
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    return xf


def inv_nx2x2(X):
    """
    Invert the N 2x2 transformation matrices stored in X; a (N,2,2) array
    :param X: transformation matrices to invert, (N,2,2) array
    :return: inverse of X
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    rdet = 1.0 / (X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1])
    y = np.zeros_like(X)
    y[:, 0, 0] = X[:, 1, 1] * rdet
    y[:, 1, 1] = X[:, 0, 0] * rdet
    y[:, 0, 1] = -X[:, 0, 1] * rdet
    y[:, 1, 0] = -X[:, 1, 0] * rdet
    return y

def inv_nx2x3(m):
    """
    Invert the N 2x3 transformation matrices stored in X; a (N,2,3) array
    :param X: transformation matrices to invert, (N,2,3) array
    :return: inverse of X
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    m2 = m[:, :, :2]
    mx = m[:, :, 2:3]
    m2inv = inv_nx2x2(m2)
    mxinv = np.matmul(m2inv, -mx)
    return np.append(m2inv, mxinv, axis=2)

def cat_nx2x3_2(a, b):
    """
    Multiply the N 2x3 transformations stored in `a` with those in `b`
    :param a: transformation matrices, (N,2,3) array
    :param b: transformation matrices, (N,2,3) array
    :return: `a . b`
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    a2 = a[:, :, :2]
    b2 = b[:, :, :2]

    ax = a[:, :, 2:3]
    bx = b[:, :, 2:3]

    ab2 = np.matmul(a2, b2)
    abx = ax + np.matmul(a2, bx)
    return np.append(ab2, abx, axis=2)

def cat_nx2x3(*x):
    """
    Multiply the N 2x3 transformations stored in the arrays in `x`
    :param x: transformation matrices, tuple of (N,2,3) arrays
    :return: `x[0] . x[1] . ... . x[N-1]`
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    y = x[0]
    for i in range(1, len(x)):
        y = cat_nx2x3_2(y, x[i])
    return y

def translation_matrices(xlats_xy):
    """
    Generate translation matrices
    :param xlats_xy: translations as an (N, 2) array (x,y)
    :return: translations matrices, (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    N = len(xlats_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = xf[:, 1, 1] = 1.0
    xf[:, :, 2] = xlats_xy
    return xf

def scale_matrices(scale_xy):
    """
    Generate translation matrices
    :param scale_xy: scale factors as an (N, 2) array (x,y)
    :return: translations matrices, (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    N = len(scale_xy)
    xf = np.zeros((N, 2, 3), dtype=np.float32)
    xf[:, 0, 0] = scale_xy[:, 0]
    xf[:, 1, 1] = scale_xy[:, 1]
    return xf

def rotation_matrices(thetas):
    """
    Generate rotation matrices

    Counter-clockwise, +y points downwards

    Where s = sin(theta) and c = cos(theta)

    M = [[ c   s   0 ]
         [ -s  c   0 ]]

    :param thetas: rotation angles in radians as a (N,) array
    :return: rotation matrices, (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    N = len(thetas)
    c = np.cos(thetas)
    s = np.sin(thetas)
    rot_xf = np.zeros((N, 2, 3), dtype=np.float32)
    rot_xf[:, 0, 0] = rot_xf[:, 1, 1] = c
    rot_xf[:, 1, 0] = -s
    rot_xf[:, 0, 1] = s
    return rot_xf

def flip_xyd_matrices(flip_flags_xyd, image_size):
    """
    Generate flip matrices in OpenCV compatible form. Each sample has three flags: `x`, `y` and `d`:
    `x == True` -> flip horizontally
    `y == True` -> flip vertically
    `d == True` -> flip diagonal or swap X and Y axes

    :param flip_flags_xyd: per sample flip flags as a (N,[x, y, d]) array
    :param image_size: image size as a `(H, w)` tuple
    :return: flip matrices, (N,2,3) array
    """
    if flip_flags_xyd.ndim != 2:
        raise ValueError('flip_flags_xyd should have 2 dimensions, not {}'.format(flip_flags_xyd.ndim))
    if flip_flags_xyd.shape[1] != 3:
        raise ValueError('flip_flags_xyd.shape[1] should be 3 dimensions, not {}'.format(flip_flags_xyd.shape[1]))

    # False -> 1, True -> -1
    flip_scale_xy = flip_flags_xyd[:, :2] * -2 + 1
    # Negative scale factors need to be combined with a translation whose value is (image_size - 1)
    # Mask the translation with the flip flags to only apply it where flipping is done
    flip_xlat_xy = flip_flags_xyd[:, :2] * (np.array(image_size[::-1]).astype(float) - 1)

    hv_flip_xf = identity_xf(len(flip_flags_xyd))

    # Diagonal flip: swap X and Y axes
    diag = flip_flags_xyd[:, 2]
    hv_flip_xf[diag] = hv_flip_xf[diag][:, ::-1, :]

    return cat_nx2x3(
        hv_flip_xf,
        translation_matrices(flip_xlat_xy),
        scale_matrices(flip_scale_xy),
    )



def centre_xf(xf, size):
    """
    Centre the transformations in `xf` around (0,0), where the current centre is assumed to be at the
    centre of an image of shape `size`
    :param xf: transformation matrices, (N,2,3) array
    :param size: image size
    :return: centred transformation matrices, (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    height, width = size

    # centre_to_zero moves the centre of the image to (0,0)
    centre_to_zero = np.zeros((1, 2, 3), dtype=np.float32)
    centre_to_zero[0, 0, 0] = centre_to_zero[0, 1, 1] = 1.0
    centre_to_zero[0, 0, 2] = -float(width) * 0.5
    centre_to_zero[0, 1, 2] = -float(height) * 0.5

    # centre_to_zero then xf
    xf_centred = cat_nx2x3(xf, centre_to_zero)

    # move (0,0) back to the centre
    xf_centred[:, 0, 2] += float(width) * 0.5
    xf_centred[:, 1, 2] += float(height) * 0.5

    return xf_centred


def cv_to_torch(mtx, dst_size, src_size=None):
    """
    Convert transformations matrices that can be used with `cv2.warpAffine` to work with PyTorch
    grid sampling.

    NOTE: `align_corners=True` should be passed to `F.affine_Grid` and `F.grid_sample` to
    correctly match OpenCV transformations.

    `cv2.warpAffine` expects a matrix that transforms an image in pixel co-ordinates.
    PyTorch `F.affine_grid` and `F.grid_sample` maps pixel locations to a [-1, 1] grid
    and transforms these sample locations, prior to sampling the image.

    :param mtx: OpenCV transformation matrices as a (N,2,3) array
    :param dst_size: the size of the output image as a `(height, width)` tuple
    :param src_size: the size of the input image as a `(height, width)` tuple, or None to use `dst_size`
    :return: PyTorch transformation matrices as a (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    dst_scale_x = float(dst_size[1] - 1) / 2.0
    dst_scale_y = float(dst_size[0] - 1) / 2.0

    if src_size is not None:
        src_scale_x = float(src_size[1] - 1) / 2.0
        src_scale_y = float(src_size[0] - 1) / 2.0
    else:
        src_scale_x = dst_scale_x
        src_scale_y = dst_scale_y

    N = len(mtx)

    # OpenCV transforms the image, whereas the PyTorch transforms the points at which the
    # image is samples. We account for this by inverting the transformation matrices
    mtx = inv_nx2x3(mtx)

    torch_cv = identity_xf(N)
    torch_cv[:, 0, 0] = dst_scale_x
    torch_cv[:, 1, 1] = dst_scale_y
    torch_cv[:, 0, 2] = dst_scale_x
    torch_cv[:, 1, 2] = dst_scale_y

    cv_torch = identity_xf(N)
    cv_torch[:, 0, 0] = 1.0 / src_scale_x
    cv_torch[:, 1, 1] = 1.0 / src_scale_y
    cv_torch[:, 0, 2] = -1.0
    cv_torch[:, 1, 2] = -1.0

    # Transform torch co-ordinates to OpenCV, apply the transformation then OpenCV co-ordinates back to torch
    return cat_nx2x3(cv_torch, mtx, torch_cv)


def pil_to_torch(mtx, dst_size, src_size=None, align_corners=True):
    """
    Convert affine transformations matrices that can be used with Pillow `Image.transform` to work with PyTorch
    grid sampling.

    `Image.transform` expects a matrix that transforms an image in pixel co-ordinates, where pixel [0,0]
    is centred at [0.5, 0.5].
    PyTorch `F.affine_grid` and `F.grid_sample` maps pixel locations to a [-1, 1] grid
    and transforms these sample locations, prior to sampling the image.

    :param mtx: PIL transformation matrices as a (N,2,3) array
    :param dst_size: the size of the output image as a `(height, width)` tuple
    :param src_size: the size of the input image as a `(height, width)` tuple, or None to use `dst_size`
    :param align_corners: if you want to use `align_corners=False` for PyTorch `F.affine_grid` and `F.grid_sample`,
        pass `align_corners=False` here
    :return: PyTorch transformation matrices as a (N,2,3) array
    """
    # Taken from https://github.com/Britefury/pytorch-mask-rcnn/blob/refactor/maskrcnn/utils/affine_transforms.py
    if align_corners:
        dst_size = (dst_size[0] - 1, dst_size[1] - 1)
    dst_scale_x = float(dst_size[1]) / 2.0
    dst_scale_y = float(dst_size[0]) / 2.0

    if src_size is not None:
        if align_corners:
            src_size = (src_size[0] - 1, src_size[1] - 1)
        src_scale_x = float(src_size[1]) / 2.0
        src_scale_y = float(src_size[0]) / 2.0
    else:
        src_scale_x = dst_scale_x
        src_scale_y = dst_scale_y

    N = len(mtx)

    torch_cv = identity_xf(N)
    torch_cv[:, 0, 0] = dst_scale_x
    torch_cv[:, 1, 1] = dst_scale_y
    torch_cv[:, 0, 2] = dst_scale_x
    torch_cv[:, 1, 2] = dst_scale_y
    if align_corners:
        torch_cv[:, 0, 2] += 0.5
        torch_cv[:, 1, 2] += 0.5

    cv_torch = identity_xf(N)
    cv_torch[:, 0, 0] = 1.0 / src_scale_x
    cv_torch[:, 1, 1] = 1.0 / src_scale_y
    cv_torch[:, 0, 2] = -1.0
    cv_torch[:, 1, 2] = -1.0
    if align_corners:
        cv_torch[:, 0, 2] += -0.5 / src_scale_x
        cv_torch[:, 1, 2] += -0.5 / src_scale_y

    # Transform torch co-ordinates to OpenCV, apply the transformation then OpenCV co-ordinates back to torch
    return cat_nx2x3(cv_torch, mtx, torch_cv)
