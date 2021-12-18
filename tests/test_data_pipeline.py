"""
Unit tests for data pipeline.

We test the data pipeline here as getting augmentation based consistency regularization working
correctly can be rather tricky. It involves applying two stochastic transformations to an image,
resulting in two augmented images. The predictions are generated for each image, after
which consistency loss measures the difference between these predictions.
Given that the augmentations involve geometric transformations, the predictions must be
transformed in order to align them with one another.

The way in which the image is augmented depends on the transform, e.g. random crop, random scale+crop
a la Hung et al., etc.
- In general, we modify the image and the labels using OpenCV functions.
- Prediction alignment is done using the PyTorch grid sampler
- We must therefore generate an affine transformation matrix for any OpenCV operation (resize, etc.) that we
  perform
- OpenCV affine transformations use pixel co-ordinates and have the origin at the top-left,
  whereas PyTorch affine transformations operate on an image stretched over a -1.0 to 1.0 grid
  and place the origin at the centre. This further complicates applying transformations to non-square
  images as e.g. a rotation will develop a shear if it is not modified correctly.
  We therefore use the `affine.cv_to_torch` function that applies the relevant transformations to
  an OpenCV transformation to make it operate in PyTorch grid co-ordinates.

If this isn't done correctly incorrect transformations will result in non-aligned predictions and
bad consistency loss.
"""
from unittest import mock
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.util import img_as_float
import cv2
import torch, torch.nn.functional as F

import datapipe.seg_transforms_cv
from datapipe import affine, seg_transforms
from datapipe import pascal_voc_dataset

_TEST_IMAGE = None
_PASCAL_IMAGE = None
REPEATS = 10

def _make_noise_test_image():
    """A simple function that generates a smoothed random noise test image
    """
    global _TEST_IMAGE
    if _TEST_IMAGE is None:
        rng = np.random.RandomState(12345)
        noise = rng.normal(size=(256, 256, 3))
        noise = gaussian_filter(noise, (4, 4, 0))
        noise -= noise.min(axis=(0,1), keepdims=True)
        noise /= noise.max(axis=(0,1), keepdims=True)
        noise = (noise * 255).astype(np.uint8)
        _TEST_IMAGE = noise
    return _TEST_IMAGE

def _get_pascal_image0():
    global _PASCAL_IMAGE
    if _PASCAL_IMAGE is None:
        ds = pascal_voc_dataset.PascalVOCDataSource()
        _PASCAL_IMAGE = np.array(ds.x[0])
    return _PASCAL_IMAGE


def _make_test_seg_image(image):
    """Wrap an image in a SegImage. Generate labels and mask.
    """
    labels = np.argmax(image, axis=2)
    mask = np.full(image.shape[:2], fill_value=255, dtype=np.uint8)
    return seg_transforms.SegImage(image=image, labels=labels, mask=mask, xf=affine.identity_xf(1))


def paired_transform_ocv(image, transform, output_size, block_size=(1, 1)):
    transform = seg_transforms.SegTransformCompose([transform, datapipe.seg_transforms_cv.SegCVTransformNormalizeToTensor(None, None)])
    pipe = seg_transforms.SegDataPipeline(block_size, transform)

    x0, m0, xf0, x1, m1, xf1 = pipe.prepare_unsupervised_paired_batch([image])

    x0 = x0[0].transpose(1, 2, 0)
    m0 = m0[0, 0]
    x1 = x1[0].transpose(1, 2, 0)
    m1 = m1[0, 0]

    xf0_to_1 = affine.cat_nx2x3(xf1, affine.inv_nx2x3(xf0))

    image_f = img_as_float(image).astype(np.float32)

    cv0 = cv2.warpAffine(image_f, xf0[0], output_size[::-1])
    cv1 = cv2.warpAffine(image_f, xf1[0], output_size[::-1])
    x01 = cv2.warpAffine(x0, xf0_to_1[0], output_size[::-1])
    m01 = cv2.warpAffine(m0, xf0_to_1[0], output_size[::-1]) * m1

    return dict(x0=x0, cv0=cv0, x1=x1, cv1=cv1, x01=x01, m01=m01)


def _paired_transform_ocv_test(image, transform, output_size, block_size=(1, 1), atol=0.3, rtol=1e-2):
    res = paired_transform_ocv(image, transform, output_size, block_size=block_size)

    x0 = res['x0']
    cv0 = res['cv0']
    x1 = res['x1']
    cv1 = res['cv1']
    x01 = res['x01']
    m01 = res['m01']

    # cmp_image0_x0 = np.allclose(cv0, x0, atol=atol, rtol=rtol)
    # cmp_image1_x1 = np.allclose(cv1, x1, atol=atol, rtol=rtol)
    # delta_x01_x1 = np.abs(x01 * (m01[:, :, None]==1) - x1 * (m01[:, :, None]==1))
    # print('hist={}'.format(np.histogram(delta_x01_x1, bins=100)))
    # cmp_x01_x1 = np.allclose(x01 * (m01[:, :, None]==1), x1 * (m01[:, :, None]==1), atol=atol, rtol=rtol)
    #
    # print(np.abs(x0 - cv0).max())
    # print(np.abs(x1 - cv1).max())
    # print(np.abs(x01 * (m01[:, :, None]==1) - x1 * (m01[:, :, None]==1)).max())
    #
    # if m01.mean() < 0.25 or not cmp_image0_x0 or not cmp_image1_x1 or not cmp_x01_x1:
    #     from matplotlib import pyplot as plt
    #     plt.figure(figsize=(18, 18))
    #     plt.subplot(3, 3, 1)
    #     plt.title('image0')
    #     plt.imshow(cv0)
    #     plt.subplot(3, 3, 2)
    #     plt.title('image1')
    #     plt.imshow(cv1)
    #     plt.subplot(3, 3, 3)
    #     plt.title('x1 == x01 = {}'.format(cmp_x01_x1))
    #     plt.imshow(np.abs(x1 - x01) * m01[:, :, None])
    #     plt.subplot(3, 3, 4)
    #     plt.title('x0')
    #     plt.imshow(x0)
    #     plt.subplot(3, 3, 5)
    #     plt.title('x1')
    #     plt.imshow(x1)
    #     plt.subplot(3, 3, 6)
    #     plt.title('x01')
    #     plt.imshow(x01)
    #     plt.subplot(3, 3, 7)
    #     plt.title('image0 == x0 ={}'.format(cmp_image0_x0))
    #     plt.imshow(np.abs(x0 - cv0))
    #     plt.subplot(3, 3, 8)
    #     plt.title('image1 == x1 = {}'.format(cmp_image1_x1))
    #     plt.imshow(np.abs(x1 - cv1))
    #     plt.show()

    assert m01.mean() > 0.25
    assert np.allclose(cv0, x0, atol=atol, rtol=rtol)
    assert np.allclose(cv1, x1, atol=atol, rtol=rtol)
    assert np.allclose(x01 * (m01[:, :, None]==1), x1 * (m01[:, :, None]==1), atol=atol, rtol=rtol)


def paired_transform_torch(image, transform, output_size, block_size=(1, 1)):
    transform = seg_transforms.SegTransformCompose([transform, datapipe.seg_transforms_cv.SegCVTransformNormalizeToTensor(None, None)])
    pipe = seg_transforms.SegDataPipeline(block_size, transform)

    torch_device = torch.device('cpu')

    x0, m0, xf0, x1, m1, xf1 = pipe.prepare_unsupervised_paired_batch([image])

    padded_shape = x0.shape[2:4]

    xf0_to_1 = affine.cat_nx2x3(xf1, affine.inv_nx2x3(xf0))

    t_image_xf0 = affine.cv_to_torch(xf0, padded_shape, image.shape[:2])
    t_image_xf1 = affine.cv_to_torch(xf1, padded_shape, image.shape[:2])
    t_xf0_to_1 = affine.cv_to_torch(xf0_to_1, padded_shape)

    image_f = img_as_float(image).astype(np.float32)

    t_image = torch.tensor(image_f.transpose(2, 0, 1)[None, ...], dtype=torch.float, device=torch_device)
    t_x0 = torch.tensor(x0, dtype=torch.float, device=torch_device)
    t_m0 = torch.tensor(m0, dtype=torch.float, device=torch_device)
    t_m1 = torch.tensor(m1, dtype=torch.float, device=torch_device)
    t_image_xf0 = torch.tensor(t_image_xf0, dtype=torch.float, device=torch_device)
    t_image_xf1 = torch.tensor(t_image_xf1, dtype=torch.float, device=torch_device)
    t_xf0_to_1 = torch.tensor(t_xf0_to_1, dtype=torch.float, device=torch_device)

    output_shape = torch.Size(len(x0), 3, output_size[0], output_size[1])
    grid_image0 = F.affine_grid(t_image_xf0, output_shape)
    grid_image1 = F.affine_grid(t_image_xf1, output_shape)
    grid_0to1 = F.affine_grid(t_xf0_to_1, output_shape)

    t_a = F.grid_sample(t_image, grid_image0)
    t_b = F.grid_sample(t_image, grid_image1)
    t_x01 = F.grid_sample(t_x0, grid_0to1)
    t_m01 = F.grid_sample(t_m0, grid_0to1) * t_m1

    t_a_np = t_a.detach().cpu().numpy()[0].transpose(1, 2, 0)
    t_b_np = t_b.detach().cpu().numpy()[0].transpose(1, 2, 0)
    t_x01_np = t_x01.detach().cpu().numpy()[0].transpose(1, 2, 0)
    t_m01_np = t_m01.detach().cpu().numpy()[0].transpose(1, 2, 0)

    x0 = x0[0].transpose(1, 2, 0)
    x1 = x1[0].transpose(1, 2, 0)

    return dict(x0=x0, torch0=t_a_np, x1=x1, torch1=t_b_np, x01=t_x01_np, m01=t_m01_np[:, :, 0])


def _paired_transform_torch_test(image, transform, output_size, block_size=(1, 1), atol=0.3, rtol=1e-2):
    res = paired_transform_torch(image, transform, output_size, block_size)

    x0 = res['x0']
    torch0 = res['torch0']
    x1 = res['x1']
    torch1 = res['torch1']
    x01 = res['x01']
    m01 = res['m01']

    assert m01.mean() > 0.25
    assert np.allclose(torch0, x0, atol=atol, rtol=rtol)
    assert np.allclose(torch1, x1, atol=atol, rtol=rtol)
    assert np.allclose(x01 * (m01==1), x1 * (m01==1), atol=atol, rtol=rtol)



def test_SegImageTransformPad_single():
    """Test SegImageTransformPad that pads an image.
    """
    seg_img = _make_test_seg_image(_make_noise_test_image())

    img_transform = datapipe.seg_transforms_cv.SegCVTransformPad()

    # No padding required
    seg_img_padded = img_transform.pad_single(seg_img, (128, 128))
    assert seg_img_padded.image.shape == seg_img.image.shape
    assert (seg_img_padded.image == seg_img.image).all()
    assert (seg_img_padded.labels == seg_img.labels).all()
    assert (seg_img_padded.mask == seg_img.mask).all()
    assert (seg_img_padded.xf == affine.identity_xf(1)).all()

    # 384,256, pad with 64 in y
    seg_img_padded = img_transform.pad_single(seg_img, (384, 256))
    assert seg_img_padded.image.shape[:2] == (384, 256)
    assert (seg_img_padded.image[:,:,:3] == np.pad(seg_img.image, [[64,64], [0,0], [0,0]], mode='constant')).all()
    alpha = np.full((256,256), 255)
    assert (seg_img_padded.image[:,:,3] == np.pad(alpha, [[64,64], [0,0]], mode='constant')).all()
    assert (seg_img_padded.labels == np.pad(seg_img.labels, [[64,64], [0,0]], mode='constant', constant_values=255)).all()
    assert (seg_img_padded.mask == np.pad(seg_img.mask, [[64,64], [0,0]], mode='constant', constant_values=0)).all()
    assert (seg_img_padded.xf == affine.translation_matrices(np.array([[0, 64]]))).all()

    # 385,256, pad with 65,64 in y
    seg_img_padded = img_transform.pad_single(seg_img, (385, 256))
    assert seg_img_padded.image.shape[:2] == (385, 256)
    assert (seg_img_padded.image[:,:,:3] == np.pad(seg_img.image, [[64,65], [0,0], [0,0]], mode='constant')).all()
    alpha = np.full((256,256), 255)
    assert (seg_img_padded.image[:,:,3] == np.pad(alpha, [[64,65], [0,0]], mode='constant')).all()
    assert (seg_img_padded.labels == np.pad(seg_img.labels, [[64,65], [0,0]], mode='constant', constant_values=255)).all()
    assert (seg_img_padded.mask == np.pad(seg_img.mask, [[64,65], [0,0]], mode='constant', constant_values=0)).all()
    assert (seg_img_padded.xf == affine.translation_matrices(np.array([[0, 64]]))).all()

    # 256,384, pad with 64 in x
    seg_img_padded = img_transform.pad_single(seg_img, (256, 384))
    assert seg_img_padded.image.shape[:2] == (256, 384)
    assert (seg_img_padded.image[:,:,:3] == np.pad(seg_img.image, [[0,0], [64,64], [0,0]], mode='constant')).all()
    assert (seg_img_padded.image[:,:,3] == np.pad(alpha, [[0,0], [64,64]], mode='constant')).all()
    assert (seg_img_padded.labels == np.pad(seg_img.labels, [[0,0], [64,64]], mode='constant', constant_values=255)).all()
    assert (seg_img_padded.mask == np.pad(seg_img.mask, [[0,0], [64,64]], mode='constant', constant_values=0)).all()
    assert (seg_img_padded.xf == affine.translation_matrices(np.array([[64, 0]]))).all()

    # 256,385, pad with 65,64 in x
    seg_img_padded = img_transform.pad_single(seg_img, (256, 385))
    assert seg_img_padded.image.shape[:2] == (256, 385)
    assert (seg_img_padded.image[:,:,:3] == np.pad(seg_img.image, [[0,0], [64,65], [0,0]], mode='constant')).all()
    assert (seg_img_padded.image[:,:,3] == np.pad(alpha, [[0,0], [64,65]], mode='constant')).all()
    assert (seg_img_padded.labels == np.pad(seg_img.labels, [[0,0], [64,65]], mode='constant', constant_values=255)).all()
    assert (seg_img_padded.mask == np.pad(seg_img.mask, [[0,0], [64,65]], mode='constant', constant_values=0)).all()
    assert (seg_img_padded.xf == affine.translation_matrices(np.array([[64, 0]]))).all()


def test_SegImageTransformPad_pair():
    """Test SegImageTransformPad that pads an image, while operating on paired images.
    """
    seg_img0 = _make_test_seg_image(_make_noise_test_image())
    seg_img1 = _make_test_seg_image(_make_noise_test_image())

    img_transform = datapipe.seg_transforms_cv.SegCVTransformPad()

    # No padding required
    seg_img0_padded, seg_img1_padded = img_transform.pad_pair(seg_img0, seg_img1, (128, 128))
    assert (seg_img0_padded.image == seg_img1_padded.image).all()
    assert (seg_img0_padded.labels == seg_img1_padded.labels).all()
    assert (seg_img0_padded.mask == seg_img1_padded.mask).all()
    assert (seg_img0_padded.xf == seg_img1_padded.xf).all()

    # 384,256, pad with 64 in y
    seg_img0_padded, seg_img1_padded = img_transform.pad_pair(seg_img0, seg_img1, (384, 256))
    assert (seg_img0_padded.image == seg_img1_padded.image).all()
    assert (seg_img0_padded.labels == seg_img1_padded.labels).all()
    assert (seg_img0_padded.mask == seg_img1_padded.mask).all()
    assert (seg_img0_padded.xf == seg_img1_padded.xf).all()

    # 384,256, pad with 64 in x
    seg_img0_padded, seg_img1_padded = img_transform.pad_pair(seg_img0, seg_img1, (256, 384))
    assert (seg_img0_padded.image == seg_img1_padded.image).all()
    assert (seg_img0_padded.labels == seg_img1_padded.labels).all()
    assert (seg_img0_padded.mask == seg_img1_padded.mask).all()
    assert (seg_img0_padded.xf == seg_img1_padded.xf).all()


def test_SegImageNormalizeToTensor():
    seg_img0 = _make_test_seg_image(_make_noise_test_image())
    seg_img1 = _make_test_seg_image(_make_noise_test_image())

    img_transform = datapipe.seg_transforms_cv.SegCVTransformNormalizeToTensor(None, None)

    # Apply to single image
    seg_tens0 = img_transform.transform_single(seg_img0)
    assert (seg_tens0.image == img_as_float(seg_img0.image).transpose(2, 0, 1).astype(np.float32)).all()
    assert (seg_tens0.labels == seg_img0.labels[None, ...]).all()
    assert (seg_tens0.mask == img_as_float(seg_img0.mask)[None, ...].astype(np.float32)).all()
    assert (seg_tens0.xf == seg_img0.xf).all()

    # Apply to pair
    seg_tens0, seg_tens1 = img_transform.transform_pair(seg_img0, seg_img1)
    assert (seg_tens0.image == img_as_float(seg_img0.image).transpose(2, 0, 1).astype(np.float32)).all()
    assert (seg_tens0.labels == seg_img0.labels[None, ...]).all()
    assert (seg_tens0.mask == img_as_float(seg_img0.mask)[None, ...].astype(np.float32)).all()
    assert (seg_tens0.xf == seg_img0.xf).all()

    assert (seg_tens1.image == img_as_float(seg_img1.image).transpose(2, 0, 1).astype(np.float32)).all()
    assert (seg_tens1.labels == seg_img1.labels[None, ...]).all()
    assert (seg_tens1.mask == img_as_float(seg_img1.mask)[None, ...].astype(np.float32)).all()
    assert (seg_tens1.xf == seg_img1.xf).all()


def test_SegImageRandomCrop():
    seg_img0 = _make_test_seg_image(_make_noise_test_image())
    seg_img1 = _make_test_seg_image(_make_noise_test_image())

    rng = mock.Mock()
    rng.uniform = mock.Mock()

    img_transform = datapipe.seg_transforms_cv.SegCVTransformRandomCrop(crop_size=(128, 128), crop_offset=(16, 16), rng=rng)

    # Apply to single image
    rng.uniform.side_effect = [np.array([0.5, 0.5])]
    crop0 = img_transform.transform_single(seg_img0)
    assert (crop0.image == seg_img0.image[64:192, 64:192, :]).all()
    assert (crop0.labels == seg_img0.labels[64:192, 64:192]).all()
    assert (crop0.mask == seg_img0.mask[64:192, 64:192]).all()
    assert (crop0.xf == affine.translation_matrices(np.array([[-64, -64]]))).all()

    rng.uniform.side_effect = [np.array([0.25, 0.75])]
    crop0 = img_transform.transform_single(seg_img0)
    assert (crop0.image == seg_img0.image[32:160, 96:224, :]).all()
    assert (crop0.labels == seg_img0.labels[32:160, 96:224]).all()
    assert (crop0.mask == seg_img0.mask[32:160, 96:224]).all()
    assert (crop0.xf == affine.translation_matrices(np.array([[-96, -32]]))).all()

    # Apply to paired image
    # First crop at 64,64 (0.5*128), with the second at 56,72
    rng.uniform.side_effect = [np.array([0.5, 0.5]), np.array([-0.5, 0.5])]
    crop0, crop1 = img_transform.transform_pair(seg_img0, seg_img1)
    assert (crop0.image == seg_img0.image[64:192, 64:192, :]).all()
    assert (crop0.labels == seg_img0.labels[64:192, 64:192]).all()
    assert (crop0.mask == seg_img0.mask[64:192, 64:192]).all()
    assert (crop0.xf == affine.translation_matrices(np.array([[-64, -64]]))).all()

    assert (crop1.image == seg_img0.image[56:184, 72:200, :]).all()
    assert (crop1.labels == seg_img0.labels[56:184, 72:200]).all()
    assert (crop1.mask == seg_img0.mask[56:184, 72:200]).all()
    assert (crop1.xf == affine.translation_matrices(np.array([[-72, -56]]))).all()

    rng.uniform.side_effect = [np.array([0.5, 0.5]), np.array([-0.5, 0.5])]
    img_transform = datapipe.seg_transforms_cv.SegCVTransformRandomCrop(crop_size=(128, 128), crop_offset=(16, 16),
                                                                        rng=np.random.RandomState(12345))
    for _ in range(REPEATS):
        _paired_transform_ocv_test(seg_img0.image, img_transform, (128, 128), (1, 1))
        _paired_transform_torch_test(seg_img0.image, img_transform, (128, 128), (1, 1))


def test_SegImageRandomCropScaleHung():
    seg_img0 = _make_test_seg_image(_make_noise_test_image())
    seg_img1 = _make_test_seg_image(_make_noise_test_image())

    rng = mock.Mock()
    rng.randint = mock.Mock()
    rng.uniform = mock.Mock()

    img_transform = datapipe.seg_transforms_cv.SegCVTransformRandomCropScaleHung(crop_size=(128, 128), crop_offset=(16, 16), rng=rng)

    # Apply to single image
    rng.randint.side_effect = [3]   # 3 / 10 + 0.5 = scale factor of 0.8; crop of 160x160 scaled to 128x128
    rng.uniform.side_effect = [np.array([0.5, 0.5])]
    crop0 = img_transform.transform_single(seg_img0)
    assert (crop0.image == cv2.resize(seg_img0.image[48:208, 48:208, :], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert (crop0.labels == cv2.resize(seg_img0.labels[48:208, 48:208], (128, 128), interpolation=cv2.INTER_NEAREST)).all()
    assert (crop0.mask == cv2.resize(seg_img0.mask[48:208, 48:208], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert np.allclose(crop0.xf, np.array([[[0.8, 0.0, -38.4],
                                            [0.0, 0.8, -38.4]]]))

    rng.randint.side_effect = [7]   # 7 / 10 + 0.5 = scale factor of 1.2; crop of 107x107 scaled to 128x128
    rng.uniform.side_effect = [np.array([0.25, 0.75])]
    crop0 = img_transform.transform_single(seg_img0)
    assert (crop0.image == cv2.resize(seg_img0.image[37:144, 112:219, :], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert (crop0.labels == cv2.resize(seg_img0.labels[37:144, 112:219], (128, 128), interpolation=cv2.INTER_NEAREST)).all()
    assert (crop0.mask == cv2.resize(seg_img0.mask[37:144, 112:219], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert np.allclose(crop0.xf, np.array([[[128/107, 0.0, -112 * 128/107],
                                            [0.0, 128/107, -37 * 128/107]]]))

    # Apply to paired image
    # First crop at 64,64 (0.5*128), with the second at 56,72
    rng.randint.side_effect = [3]   # 3 / 10 + 0.5 = scale factor of 0.8; crop of 160x160 scaled to 128x128
    rng.uniform.side_effect = [np.array([0.5, 0.5]), np.array([-0.5, 0.5])]
    # extra=(96, 96), pos0=(48,48), pos1=(40,56), centre0=(128,128), centre1=(120,136), pos0=(64,64), pos1=(40,56)
    crop0, crop1 = img_transform.transform_pair(seg_img0, seg_img1)
    assert (crop0.image == seg_img0.image[64:192, 64:192, :]).all()
    assert (crop0.labels == seg_img0.labels[64:192, 64:192]).all()
    assert (crop0.mask == seg_img0.mask[64:192, 64:192]).all()
    assert (crop0.xf == affine.translation_matrices(np.array([[-64, -64]]))).all()

    assert (crop1.image == cv2.resize(seg_img1.image[40:200, 56:216, :], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert (crop1.labels == cv2.resize(seg_img1.labels[40:200, 56:216], (128, 128), interpolation=cv2.INTER_NEAREST)).all()
    assert (crop1.mask == cv2.resize(seg_img1.mask[40:200, 56:216], (128, 128), interpolation=cv2.INTER_LINEAR)).all()
    assert np.allclose(crop1.xf, np.array([[[0.8, 0.0, -44.8],
                                            [0.0, 0.8, -32.0]]]))
    #
    # pascal_image = _get_pascal_image0()
    # img_transform = data_pipeline.SegImageRandomCropScaleHung(crop_size=(128, 128), crop_offset=(16, 16),
    #                                                           rng=np.random.RandomState(12345))
    # for _ in range(REPEATS):
    #     _paired_transform_ocv_test(pascal_image, img_transform, (128, 128), (1, 1))
    #     _paired_transform_torch_test(pascal_image, img_transform, (128, 128), (1, 1))


def test_SegImageRandomFlip():
    seg_img0 = _make_test_seg_image(_make_noise_test_image())
    seg_img1 = _make_test_seg_image(_make_noise_test_image())

    rng = mock.Mock()
    rng.uniform = mock.Mock()

    img_transform = datapipe.seg_transforms_cv.SegCVTransformRandomFlip(hflip=True, vflip=True, hvflip=True, rng=rng)

    # Apply to single image
    rng.binomial.side_effect = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    hflipped = img_transform.transform_single(seg_img0)
    assert (hflipped.image == seg_img0.image[:, ::-1, :]).all()
    assert (hflipped.labels == seg_img0.labels[:, ::-1]).all()
    assert (hflipped.mask == seg_img0.mask[:, ::-1]).all()
    assert (hflipped.xf == np.array([[[-1, 0, 255.0],
                                      [0,  1, 0]]])).all()

    vflipped = img_transform.transform_single(seg_img0)
    assert (vflipped.image == seg_img0.image[::-1, :, :]).all()
    assert (vflipped.labels == seg_img0.labels[::-1, :]).all()
    assert (vflipped.mask == seg_img0.mask[::-1, :]).all()
    assert (vflipped.xf == np.array([[[1, 0, 0],
                                      [0, -1, 255.0]]])).all()

    hvflipped = img_transform.transform_single(seg_img0)
    assert (hvflipped.image == seg_img0.image.transpose(1, 0, 2)).all()
    assert (hvflipped.labels == seg_img0.labels.T).all()
    assert (hvflipped.mask == seg_img0.mask.T).all()
    assert (hvflipped.xf == np.array([[[0, 1, 0],
                                       [1, 0, 0]]])).all()

    img_transform = datapipe.seg_transforms_cv.SegCVTransformRandomFlip(hflip=True, vflip=True, hvflip=True,
                                                                        rng=np.random.RandomState(12345))
    for _ in range(REPEATS):
        _paired_transform_ocv_test(seg_img0.image, img_transform, (256, 256), (1, 1))
        _paired_transform_torch_test(seg_img0.image, img_transform, (256, 256), (1, 1))


def test_SegDataPipeline():
    seg_img0 = _make_test_seg_image(_make_noise_test_image())

    rng = mock.Mock()
    rng.uniform = mock.Mock()

    img_transform = seg_transforms.SegTransformCompose([
        datapipe.seg_transforms_cv.SegCVTransformRandomCrop(crop_size=(128, 128), crop_offset=(16, 16), rng=rng),
        datapipe.seg_transforms_cv.SegCVTransformNormalizeToTensor(None, None),
    ])
    pipe = seg_transforms.SegDataPipeline((32, 32), img_transform)

    # Supervised batch
    rng.uniform.side_effect = [np.array([0.5, 0.5])]
    xs, ys = pipe.prepare_supervised_batch([seg_img0.image], [seg_img0.labels])
    assert (xs == img_as_float(seg_img0.image).transpose(2, 0, 1)[None, :, 64:192, 64:192].astype(np.float32)).all()
    assert (ys == seg_img0.labels[None, None, 64:192, 64:192]).all()

    # Unsupervised batch
    rng.uniform.side_effect = [np.array([0.5, 0.5])]
    xs, ms = pipe.prepare_unsupervised_batch([seg_img0.image])
    assert (xs == img_as_float(seg_img0.image).transpose(2, 0, 1)[None, :, 64:192, 64:192].astype(np.float32)).all()
    assert (ms == np.ones((1, 1, 128, 128), dtype=np.float32)).all()

    # Unsupervised paired batch
    rng.uniform.side_effect = [np.array([0.5, 0.5]), np.array([-0.5, 0.5])]
    x0s, m0s, xf0s, x1s, m1s, xf1s = pipe.prepare_unsupervised_paired_batch([seg_img0.image])
    assert (x0s == img_as_float(seg_img0.image).transpose(2, 0, 1)[None, :, 64:192, 64:192].astype(np.float32)).all()
    assert (x1s == img_as_float(seg_img0.image).transpose(2, 0, 1)[None, :, 56:184, 72:200].astype(np.float32)).all()
    assert (m0s == np.ones((1, 1, 128, 128), dtype=np.float32)).all()
    assert (m1s == np.ones((1, 1, 128, 128), dtype=np.float32)).all()
    assert (xf0s == affine.translation_matrices(np.array([[-64, -64]]))).all()
    assert (xf1s == affine.translation_matrices(np.array([[-72, -56]]))).all()
