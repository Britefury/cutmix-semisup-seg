import numpy as np
from scipy.signal import fftconvolve


def neighbouring_pixels_class_change(y):
    """
    Compute 4 2D boolean arrays that indicate for each pixel if the neighbour to the
    left, right, above or below (respectively) have a different value/class.

    :param y: 2D image as a (H,W) NumPy array
    :return: `(left, right, up, down)` where each element is a (H, W) NumPy array
        that indicates if the neighbouring pixel in those directions have a different value
        in `y`
    """
    y_cen = y[1:-1, 1:-1]
    left = (y_cen != y[1:-1, :-2]) & (y[1:-1, :-2] != 255)
    right = (y_cen != y[1:-1, 2:]) & (y[1:-1, 2:] != 255)
    up = (y_cen != y[:-2, 1:-1]) & (y[:-2, 1:-1] != 255)
    down = (y_cen != y[2:, 1:-1]) & (y[2:, 1:-1] != 255)
    valid = (y_cen != 255)
    return (np.pad(valid & left, [[1, 1], [1, 1]], mode='constant'),
            np.pad(valid & right, [[1, 1], [1, 1]], mode='constant'),
            np.pad(valid & up, [[1, 1], [1, 1]], mode='constant'),
            np.pad(valid & down, [[1, 1], [1, 1]], mode='constant'))


def boundary_pixels(y):
    """
    Compute a 2D boolean array that indicates if a pixel is adjacent to one or more
    neighbours whose value in `y` differs.

    :param y: 2D image as a (H,W) NumPy array
    :return: (H, W) NumPy array with dtype=bool
    """
    left, right, up, down = neighbouring_pixels_class_change(y)
    return left | right | up | down


def box_sum(x, box_size):
    """
    Compute the sliding window sum of a box of size (sz, sz)
    :param x: image as a (H, W) NumPy array
    :param box_size: box size as a (h, w) tuple
    :return: sliding window sum as a (H+1-sz, W+1-sz) NumPy array
    """
    s = np.cumsum(np.cumsum(x, axis=1), axis=0)
    s = np.pad(s, [[1, 0], [1, 0]], mode='constant')
    d = s[box_size[0]:, box_size[1]:] - s[:-box_size[0], box_size[1]:] - \
        s[box_size[0]:, :-box_size[1]] + s[:-box_size[0], :-box_size[1]]

    # The above as the same effect as:
    # uniform_filter(x, sz, mode='constant')[pad:-pad, pad:-pad]*sz*sz

    return d


def neighbouring_patch_distance_maps(x, patch_size):
    """
    Compute 4 2D arrays that give the Euclidean distance between patches
    centred on neighbouring pixels to the left, right, above or below.

    :param y: 2D image as a (H,W,C) NumPy array
    :param patch_size: patch size as a (h, w) tuple
    :return: `(left, right, up, down)` where each element is a
        distance map as a (H, W) NumPy array with dtype=float
    """

    patch_size = np.array(patch_size)
    pad = (patch_size - 1) // 2
    x = np.pad(x, [[pad[0] + 1, pad[0] + 1], [pad[1] + 1, pad[1] + 1], [0, 0]], mode='symmetric')

    x_cen = x[1:-1, 1:-1, :]
    left_grad = x_cen - x[1:-1, :-2, :]
    right_grad = x[1:-1, 2:, :] - x_cen
    up_grad = x_cen - x[:-2, 1:-1, :]
    down_grad = x[2:, 1:-1, :] - x_cen
    left_grad_2 = (left_grad ** 2).sum(axis=2)
    right_grad_2 = (right_grad ** 2).sum(axis=2)
    up_grad_2 = (up_grad ** 2).sum(axis=2)
    down_grad_2 = (down_grad ** 2).sum(axis=2)

    left_d = np.sqrt(box_sum(left_grad_2, patch_size))
    right_d = np.sqrt(box_sum(right_grad_2, patch_size))
    up_d = np.sqrt(box_sum(up_grad_2, patch_size))
    down_d = np.sqrt(box_sum(down_grad_2, patch_size))

    return left_d, right_d, up_d, down_d


def patch_average_distance_map(x, patch_size):
    """
    Compute a 2D arrays that give the average Euclidean distance between a patch centred
    on a pixel and the 4 patches centred on neighbouring pixels to the left, right,
    above or below.

    :param y: 2D image as a (H,W,C) NumPy array
    :param patch_size: patch size as a (h, w) tuple
    :return: distance map as (H, W) NumPy array with dtype=float
    """
    left_d, right_d, up_d, down_d = neighbouring_patch_distance_maps(x, patch_size)

    avg_d = (left_d + right_d + up_d + down_d) * 0.25

    return avg_d


def sliding_window_distance_to_patch(image, patch):
    """
    Compute the Euclidean pixel distance from the image patch `patch` to all patches
    of the same size extracted from `image` in a sliding window fashion

    :param image: input image as a (H, W, C) NumPy array
    :param patch: query patch as a (p, q, C) NumPy array
    :return: distance map as a (H, W) NumPy array
    """
    patch_size = np.array(patch.shape[:2])
    pad = (patch_size - 1) // 2
    image = np.pad(image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='symmetric')

    P_sqr = box_sum((image*image).sum(axis=2), patch_size)
    Q_sqr = (patch * patch).sum()
    P_Q = 0
    for chn_i in range(image.shape[2]):
        P_Q += fftconvolve(image[:, :, chn_i], patch[::-1, ::-1, chn_i], mode='valid')

    sqr_dist = P_sqr + Q_sqr - 2*P_Q
    return np.sqrt(np.maximum(sqr_dist, 0))


def sliding_window_distance_to_patches_generator(image, patches):
    """
    Compute the Euclidean pixel distance from each of the image patches `patches` to all patches
    of the same size extracted from `image` in a sliding window fashion. Return a generator
    that yields one distance map for each patch in `patches`.

    :param image: input image as a (H, W, C) NumPy array
    :param patches: query patch as a (N, p, q, C) NumPy array
    :return: generator that yields distance maps as (N, H, W) NumPy arrays
    """
    patch_size = np.array(patches.shape[1:3])
    pad = (patch_size - 1) // 2
    image = np.pad(image, [[pad[0], pad[0]], [pad[1], pad[1]], [0, 0]], mode='symmetric')

    P_sqr = box_sum((image*image).sum(axis=2), patch_size)
    Q_sqr = (patches * patches).sum(axis=(1,2,3))
    dist_maps = []
    for patch_i in range(patches.shape[0]):
        P_Q = 0
        for chn_i in range(image.shape[2]):
            P_Q += fftconvolve(image[:, :, chn_i], patches[patch_i, ::-1, ::-1, chn_i], mode='valid')

        sqr_dist = P_sqr + Q_sqr[patch_i] - 2*P_Q
        dist_map = np.sqrt(np.maximum(sqr_dist, 0))
        yield dist_map


def extract_patch(image, patch_shape, yx):
    """
    Extract a patch of shape `patch_shape` from the image `image` centred a position `yx`
    :param image: image as a `(H, W, C)` NumPy array
    :param patch_shape: patch shape as [height, width] as list, tuple or NumPy array
    :param yx: patch centre as [y,x]
    :return: patch as a `(height, width, C)` NumPy array
    """
    patch_shape = np.array(patch_shape)
    pad = (patch_shape - 1) // 2
    row, col = yx
    return image[row - pad[0]:row + pad[0] + 1, col - pad[1]:col + pad[1] + 1, ...]


