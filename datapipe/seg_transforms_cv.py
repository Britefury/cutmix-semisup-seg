"""
Transformations that use an OpenCV transformation pipeline.

`SegCVTransformRandomCropScaleHung` follows the scheme used in Hung et al. [1] and Mittal et al. [2]

[1] 'Adversarial Learning for Semi-Supervised Semantic Segmentation' by Hung et al.
https://arxiv.org/abs/1802.07934

[2] 'Semi-Supervised Semantic Segmentation with High- and Low-level Consistency' by Mittal et al.
https://arxiv.org/abs/1908.05724
"""

import math

import cv2
import numpy as np
from skimage import img_as_float
from PIL import Image

from datapipe import affine
from datapipe.seg_transforms import SegTransform

# PyTorch data loaders use multi-processing.
# OpenCV uses threads that are not replicated when the process is forked,
# causing OpenCV functions to lock up, so we have to tell OpenCV not to use threads
cv2.setNumThreads(0)


class SegCVTransformPad (SegTransform):
    def pad_single(self, sample, min_size):
        sample = sample.copy()
        image = sample['image_arr']
        # image, labels, mask, xf = seg_img.image, seg_img.labels, seg_img.mask, seg_img.xf
        img_size = image.shape[:2]
        if img_size[0] < min_size[0] or img_size[1] < min_size[1]:
            # Padding required

            # Compute padding
            pad_h = max(min_size[0] - img_size[0], 0)
            pad_w = max(min_size[1] - img_size[1], 0)
            h0 = pad_h // 2
            h1 = pad_h - h0
            w0 = pad_w // 2
            w1 = pad_w - w0

            # Add an alpha channel to the image so that we can use it during standardisation
            # to ensure that the padding area has a value of 0, post mean-subtraction
            alpha_channel = np.ones(img_size + (1,), dtype=image.dtype) * 255
            image = np.append(image[:, :, :3], alpha_channel, axis=2)

            # Apply
            sample['image_arr'] = np.pad(image, [[h0, h1], [w0, w1], [0, 0]], mode='constant', constant_values=0)
            if 'labels_arr' in sample:
                sample['labels_arr'] = np.pad(sample['labels_arr'], [[h0, h1], [w0, w1]], mode='constant', constant_values=255)
            if 'mask_arr' in sample:
                sample['mask_arr'] = np.pad(sample['mask_arr'], [[h0, h1], [w0, w1]], mode='constant')
            if 'xf_cv' in sample:
                sample['xf_cv'] = affine.cat_nx2x3(
                    affine.translation_matrices(np.array([[w0, h0]])),
                    sample['xf_cv'][None, ...]
                )[0]
        return sample

    def pad_pair(self, sample0, sample1, min_size):
        sample0 = sample0.copy()
        sample1 = sample1.copy()
        image0 = sample0['image_arr']
        image1 = sample1['image_arr']
        img_size0 = image0.shape[:2]
        if img_size0[0] < min_size[0] or img_size0[1] < min_size[1]:
            # Padding required

            # Compute padding
            pad_h = max(min_size[0] - img_size0[0], 0)
            pad_w = max(min_size[1] - img_size0[1], 0)
            h0 = pad_h // 2
            h1 = pad_h - h0
            w0 = pad_w // 2
            w1 = pad_w - w0

            # Add an alpha channel to the image so that we can use it during standardisation
            # to ensure that the padding area has a value of 0, post mean-subtraction
            alpha_channel = np.ones(img_size0 + (1,), dtype=image0.dtype) * 255
            image0 = np.append(image0[:, :, :3], alpha_channel, axis=2)
            image1 = np.append(image1[:, :, :3], alpha_channel, axis=2)

            # Apply
            sample0['image_arr'] = np.pad(image0, [[h0, h1], [w0, w1], [0, 0]], mode='constant', constant_values=0)
            sample1['image_arr'] = np.pad(image1, [[h0, h1], [w0, w1], [0, 0]], mode='constant', constant_values=0)
            if 'labels_arr' in sample0:
                sample0['labels_arr'] = np.pad(sample0['labels_arr'], [[h0, h1], [w0, w1]], mode='constant', constant_values=255)
                sample1['labels_arr'] = np.pad(sample1['labels_arr'], [[h0, h1], [w0, w1]], mode='constant', constant_values=255)
            if 'mask_arr' in sample0:
                sample0['mask_arr'] = np.pad(sample0['mask_arr'], [[h0, h1], [w0, w1]], mode='constant')
                sample1['mask_arr'] = np.pad(sample1['mask_arr'], [[h0, h1], [w0, w1]], mode='constant')
            if 'xf_cv' in sample0:
                pad_xlat = affine.translation_matrices(np.array([[w0, h0]]))
                sample0['xf_cv'] = affine.cat_nx2x3(pad_xlat, sample0['xf_cv'][None, ...])[0]
                sample1['xf_cv'] = affine.cat_nx2x3(pad_xlat, sample1['xf_cv'][None, ...])[0]
        return (sample0, sample1)


class SegCVTransformRandomCrop (SegCVTransformPad):
    def __init__(self, crop_size, crop_offset, rng=None):
        if crop_offset is None:
            crop_offset = [0, 0]
        self.crop_size = np.array(crop_size)
        self.crop_offset = np.array(crop_offset)
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng


    def transform_single(self, sample):
        sample = self.pad_single(sample, self.crop_size)
        image = sample['image_arr']
        extra = np.array(image.shape[:2]) - self.crop_size
        pos = np.round(extra * self.rng.uniform(0.0, 1.0, size=(2,))).astype(int)
        sample['image_arr'] = image[pos[0]:pos[0]+self.crop_size[0], pos[1]:pos[1]+self.crop_size[1]]
        if 'labels_arr' in sample:
            sample['labels_arr'] = sample['labels_arr'][pos[0]:pos[0]+self.crop_size[0], pos[1]:pos[1]+self.crop_size[1]]
        if 'mask_arr' in sample:
            sample['mask_arr'] = sample['mask_arr'][pos[0]:pos[0] + self.crop_size[0], pos[1]:pos[1] + self.crop_size[1]]
        if 'xf_cv' in sample:
            sample['xf_cv'] = affine.cat_nx2x3(
                affine.translation_matrices(-pos[None, ::-1].astype(float)),
                sample['xf_cv'][None, ...]
            )[0]
        return sample

    def transform_pair(self, sample0, sample1):
        # Pad the image if necessary
        sample0, sample1 = self.pad_pair(sample0, sample1, self.crop_size)

        # Randomly choose positions of each crop
        extra = np.array(sample0['image_arr'].shape[:2]) - self.crop_size
        pos0 = np.round(extra * self.rng.uniform(0.0, 1.0, size=(2,))).astype(int)
        pos1 = pos0 + np.round(self.crop_offset * self.rng.uniform(-1.0, 1.0, size=(2,))).astype(int)
        # Ensure pos1 cannot go out of bounds
        pos1 = np.clip(pos1, np.array([0, 0]), extra)

        # Extract crop and scale to target size
        sample0['image_arr'] = sample0['image_arr'][pos0[0]:pos0[0] + self.crop_size[0], pos0[1]:pos0[1] + self.crop_size[1]]
        sample1['image_arr'] = sample1['image_arr'][pos1[0]:pos1[0] + self.crop_size[0], pos1[1]:pos1[1] + self.crop_size[1]]

        sample0['mask_arr'] = sample0['mask_arr'][pos0[0]:pos0[0] + self.crop_size[0], pos0[1]:pos0[1] + self.crop_size[1]]
        sample1['mask_arr'] = sample1['mask_arr'][pos1[0]:pos1[0] + self.crop_size[0], pos1[1]:pos1[1] + self.crop_size[1]]

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = sample0['labels_arr'][pos0[0]:pos0[0] + self.crop_size[0], pos0[1]:pos0[1] + self.crop_size[1]]
            sample1['labels_arr'] = sample1['labels_arr'][pos1[0]:pos1[0] + self.crop_size[0], pos1[1]:pos1[1] + self.crop_size[1]]

        if 'xf_cv' in sample0:
            sample0['xf_cv'] = affine.cat_nx2x3(
                affine.translation_matrices(-pos0[None, ::-1]),
                sample0['xf_cv'][None, ...]
            )[0]
            sample1['xf_cv'] = affine.cat_nx2x3(
                affine.translation_matrices(-pos1[None, ::-1]),
                sample1['xf_cv'][None, ...]
            )[0]
        return (sample0, sample1)


class SegCVTransformRandomCropScaleHung (SegCVTransformPad):
    """
    Random crop with random scale.
    """
    def __init__(self, crop_size, crop_offset, uniform_scale=True, rng=None):
        if crop_offset is None:
            crop_offset = [0, 0]
        self.crop_size = tuple(crop_size)
        self.crop_size_arr = np.array(crop_size)
        self.crop_offset = np.array(crop_offset)
        self.uniform_scale = uniform_scale
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    def transform_single(self, sample0):
        sample0 = sample0.copy()
        scale_dim = 1 if self.uniform_scale else 2

        # Draw scale factor
        f_scale = 0.5 + self.rng.randint(0, 11, size=(scale_dim,)) / 10.0

        # Scale the crop size by the inverse of the scale
        sc_size = np.round(self.crop_size_arr / f_scale).astype(int)

        sample0 = self.pad_single(sample0, sc_size)
        image, labels, mask, xf = sample0['image_arr'], sample0.get('labels_arr'), sample0.get('mask_arr'), sample0.get('xf_cv')

        # Randomly choose position
        extra = np.array(image.shape[:2]) - sc_size
        pos = np.round(extra * self.rng.uniform(0.0, 1.0, size=(2,))).astype(int)

        # Extract crop and scale to target size
        image = image[pos[0]:pos[0]+sc_size[0], pos[1]:pos[1]+sc_size[1]]
        sample0['image_arr'] = cv2.resize(image, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)

        if labels is not None:
            labels = labels[pos[0]:pos[0]+sc_size[0], pos[1]:pos[1]+sc_size[1]]
            sample0['labels_arr'] = cv2.resize(labels, self.crop_size[::-1], interpolation = cv2.INTER_NEAREST)

        if mask is not None:
            mask = mask[pos[0]:pos[0] + sc_size[0], pos[1]:pos[1] + sc_size[1]]
            sample0['mask_arr'] = cv2.resize(mask, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)

        if xf is not None:
            # Matching `cv2.resize` requires:
            # - scale factor of out_size/in_size
            # - a translation of (scale_factor - 1) / 2
            scale_factor_yx = self.crop_size_arr / sc_size
            resize_xlat_yx = (scale_factor_yx - 1.0) * 0.5
            sample0['xf_cv'] = affine.cat_nx2x3(
                affine.translation_matrices(resize_xlat_yx[None, ::-1].astype(float)),
                affine.scale_matrices(scale_factor_yx[None, ::-1]),
                affine.translation_matrices(-pos[None, ::-1].astype(float)),
                xf[None, ...]
            )[0]

        return sample0

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        scale_dim = 1 if self.uniform_scale else 2

        # Draw a scale factor for the second crop (crop1 from crop0,crop1)
        f_scale1 = 0.5 + self.rng.randint(0, 11, size=(scale_dim,)) / 10.0

        # Scale the crop size by the inverse of the scale
        sc_size1 = np.round(self.crop_size_arr / f_scale1).astype(int)
        # Compute the maximum crop size that we need
        max_sc_size = np.maximum(self.crop_size_arr, sc_size1)

        # Pad the image if necessary
        sample0, sample1 = self.pad_pair(sample0, sample1, max_sc_size)

        # Randomly choose positions of each crop
        extra = np.array(sample0['image_arr'].shape[:2]) - max_sc_size
        pos0 = np.round(extra * self.rng.uniform(0.0, 1.0, size=(2,))).astype(int)
        pos1 = pos0 + np.round(self.crop_offset * self.rng.uniform(-1.0, 1.0, size=(2,))).astype(int)
        # Ensure pos1 cannot go out of bounds
        pos1 = np.clip(pos1, np.array([0, 0]), extra)

        centre0 = pos0 + max_sc_size * 0.5
        centre1 = pos1 + max_sc_size * 0.5

        pos0 = np.round(centre0 - self.crop_size_arr * 0.5).astype(int)
        pos1 = np.round(centre1 - sc_size1 * 0.5).astype(int)

        # Extract crop and scale to target size
        sample0['image_arr'] = sample0['image_arr'][pos0[0]:pos0[0] + self.crop_size_arr[0], pos0[1]:pos0[1] + self.crop_size_arr[1]]
        sample1['image_arr'] = sample1['image_arr'][pos1[0]:pos1[0] + sc_size1[0], pos1[1]:pos1[1] + sc_size1[1]]
        # image0 = cv2.resize(image0, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
        sample1['image_arr'] = cv2.resize(sample1['image_arr'], self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = sample0['mask_arr'][pos0[0]:pos0[0] + self.crop_size_arr[0], pos0[1]:pos0[1] + self.crop_size_arr[1]]
            sample1['mask_arr'] = sample1['mask_arr'][pos1[0]:pos1[0] + sc_size1[0], pos1[1]:pos1[1] + sc_size1[1]]
            # mask0 = cv2.resize(mask0, self.crop_size[::-1], interpolation=cv2.INTER_NEAREST)
            sample1['mask_arr'] = cv2.resize(sample1['mask_arr'], self.crop_size[::-1], interpolation=cv2.INTER_NEAREST)

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = sample0['labels_arr'][pos0[0]:pos0[0] + self.crop_size_arr[0], pos0[1]:pos0[1] + self.crop_size_arr[1]]
            sample1['labels_arr'] = sample1['labels_arr'][pos1[0]:pos1[0] + sc_size1[0], pos1[1]:pos1[1] + sc_size1[1]]
            # labels0 = cv2.resize(labels0, self.crop_size[::-1], interpolation = cv2.INTER_NEAREST)
            sample1['labels_arr'] = cv2.resize(sample1['labels_arr'], self.crop_size[::-1], interpolation = cv2.INTER_NEAREST)

        if 'xf_cv' in sample0:
            xf01 = np.stack([sample0['xf_cv'], sample1['xf_cv']], axis=0)

            positions_xy = np.append(pos0[None, ::-1], pos1[None, ::-1], axis=0)
            # Matching `cv2.resize` requires:
            # - scale factor of out_size/in_size
            # - a translation of (scale_factor - 1) / 2
            scale_factors_xy = np.append(
                np.array([[1, 1]]),
                self.crop_size_arr[None, ::-1].astype(float) / sc_size1[None, ::-1],
                axis=0
            )
            resize_xlats_xy = (scale_factors_xy - 1.0) * 0.5

            xf01 = affine.cat_nx2x3(
                affine.translation_matrices(resize_xlats_xy),
                affine.scale_matrices(scale_factors_xy),
                affine.translation_matrices(-positions_xy),
                xf01,
            )
            sample0['xf_cv'] = xf01[0]
            sample1['xf_cv'] = xf01[1]

        return sample0, sample1


class SegCVTransformRandomCropRotateScale (SegCVTransformPad):
    """
    Random crop with random scale and rotate.
    """
    def __init__(self, crop_size, crop_offset, rot_mag, max_scale, uniform_scale=True, constrain_rot_scale=True,
                 rng=None):
        if crop_offset is None:
            crop_offset = [0, 0]
        self.crop_size = tuple(crop_size)
        self.crop_size_arr = np.array(crop_size)
        self.crop_offset = np.array(crop_offset)
        self.rot_mag_rad = math.radians(rot_mag)
        self.log_max_scale = np.log(max_scale)
        self.uniform_scale = uniform_scale
        self.constrain_rot_scale = constrain_rot_scale
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    def transform_single(self, sample0):
        sample0 = sample0.copy()

        # Extract contents
        image = sample0['image_arr']

        # Choose scale and rotation
        if self.uniform_scale:
            scale_factor_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1,)))
            scale_factor_yx = np.repeat(scale_factor_yx, 2, axis=0)
        else:
            scale_factor_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2,)))
        rot_theta = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(1,))

        # Scale the crop size by the inverse of the scale
        sc_size = self.crop_size_arr / scale_factor_yx

        # Randomly choose centre
        img_size = np.array(image.shape[:2])
        extra = np.maximum(img_size - sc_size, 0.0)
        centre = extra * self.rng.uniform(0.0, 1.0, size=(2,)) + np.minimum(sc_size, img_size) * 0.5

        # Build affine transformation matrix
        local_xf = affine.cat_nx2x3(
            affine.translation_matrices(self.crop_size_arr[None, ::-1] * 0.5),
            affine.rotation_matrices(rot_theta),
            affine.scale_matrices(scale_factor_yx[None, ::-1]),
            affine.translation_matrices(-centre[None, ::-1]),
        )

        # Reflect the image
        # Use nearest neighbour sampling to stay consistent with labels, if labels present
        if 'labels_arr' in sample0:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = self.rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])

        sample0['image_arr'] = cv2.warpAffine(image, local_xf[0], self.crop_size[::-1], flags=interpolation, borderValue=0, borderMode=cv2.BORDER_REFLECT_101)

        # Don't reflect labels and mask
        if 'labels_arr' in sample0:
            sample0['labels_arr'] = cv2.warpAffine(sample0['labels_arr'], local_xf[0], self.crop_size[::-1], flags=cv2.INTER_NEAREST, borderValue=255, borderMode=cv2.BORDER_CONSTANT)

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = cv2.warpAffine(sample0['mask_arr'], local_xf[0], self.crop_size[::-1], flags=interpolation, borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'xf_cv' in sample0:
            sample0['xf_cv'] = affine.cat_nx2x3(local_xf, sample0['xf_cv'][None, ...])[0]

        return sample0

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Choose scales and rotations
        if self.constrain_rot_scale:
            if self.uniform_scale:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1, 1)))
                scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=1)
            else:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1, 2)))

            rot_thetas = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(1,))
            scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=0)
            rot_thetas = np.repeat(rot_thetas, 2, axis=0)
        else:
            if self.uniform_scale:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2, 1)))
                scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=1)
            else:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2, 2)))
            rot_thetas = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(2,))

        img_size = np.array(sample0['image_arr'].shape[:2])

        # Scale the crop size by the inverse of the scale
        sc_size = self.crop_size_arr / scale_factors_yx.min(axis=0)
        crop_centre_pos = np.minimum(sc_size, img_size) * 0.5

        # Randomly choose centres
        extra = np.maximum(img_size - sc_size, 0.0)
        centre0 = extra * self.rng.uniform(0.0, 1.0, size=(2,)) + crop_centre_pos
        offset1 = np.round(self.crop_offset * self.rng.uniform(-1.0, 1.0, size=(2,)))
        centre_xlat = np.stack([centre0, centre0], axis=0)
        offset1_xlat = np.stack([np.zeros((2,)), offset1], axis=0)

        # Build affine transformation matrices
        local_xfs = affine.cat_nx2x3(
            affine.translation_matrices(self.crop_size_arr[None, ::-1] * 0.5),
            affine.translation_matrices(offset1_xlat[:, ::-1]),
            affine.rotation_matrices(rot_thetas),
            affine.scale_matrices(scale_factors_yx[:, ::-1]),
            affine.translation_matrices(-centre_xlat[:, ::-1]),
        )

        # Use nearest neighbour sampling to stay consistent with labels, if labels present
        interpolation = cv2.INTER_NEAREST if 'labels_arr' in sample0 else cv2.INTER_LINEAR
        sample0['image_arr'] = cv2.warpAffine(sample0['image_arr'], local_xfs[0], self.crop_size[::-1], flags=interpolation,
                                          borderValue=0, borderMode=cv2.BORDER_REFLECT_101)
        sample1['image_arr'] = cv2.warpAffine(sample1['image_arr'], local_xfs[1], self.crop_size[::-1], flags=interpolation,
                                          borderValue=0, borderMode=cv2.BORDER_REFLECT_101)

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = cv2.warpAffine(sample0['labels_arr'], local_xfs[0], self.crop_size[::-1], flags=cv2.INTER_NEAREST,
                                               borderValue=255, borderMode=cv2.BORDER_CONSTANT)
            sample1['labels_arr'] = cv2.warpAffine(sample1['labels_arr'], local_xfs[1], self.crop_size[::-1], flags=cv2.INTER_NEAREST,
                                               borderValue=255, borderMode=cv2.BORDER_CONSTANT)

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = cv2.warpAffine(sample0['mask_arr'], local_xfs[0], self.crop_size[::-1], flags=interpolation,
                                             borderValue=0, borderMode=cv2.BORDER_CONSTANT)
            sample1['mask_arr'] = cv2.warpAffine(sample1['mask_arr'], local_xfs[1], self.crop_size[::-1], flags=interpolation,
                                             borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'xf_cv' in sample0:
            xf01 = affine.cat_nx2x3(local_xfs, np.stack([sample0['xf_cv'], sample1['xf_cv']], axis=0))
            sample0['xf_cv'] = xf01[0]
            sample1['xf_cv'] = xf01[1]

        return (sample0, sample1)


class SegCVTransformRandomFlip (SegTransform):
    def __init__(self, hflip, vflip, hvflip, rng=None):
        self.hflip = hflip
        self.vflip = vflip
        self.hvflip = hvflip
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    @staticmethod
    def flip_image(img, flip_xyd):
        if flip_xyd[0]:
            img = img[:, ::-1]
        if flip_xyd[1]:
            img = img[::-1, ...]
        if flip_xyd[2]:
            img = np.swapaxes(img, 0, 1)
        return img.copy()

    def transform_single(self, sample):
        sample = sample.copy()

        # Flip flags
        flip_flags_xyd = self.rng.binomial(1, 0.5, size=(3,)) != 0
        flip_flags_xyd = flip_flags_xyd & np.array([self.hflip, self.vflip, self.hvflip])

        sample['image_arr'] = self.flip_image(sample['image_arr'], flip_flags_xyd)

        if 'mask_arr' in sample:
            sample['mask_arr'] = self.flip_image(sample['mask_arr'], flip_flags_xyd)

        if 'labels_arr' in sample:
            sample['labels_arr'] = self.flip_image(sample['labels_arr'], flip_flags_xyd)

        if 'xf_cv' in sample:
            sample['xf_cv'] = affine.cat_nx2x3(
                affine.flip_xyd_matrices(flip_flags_xyd, sample['image_arr'].shape[:2]),
                sample['xf_cv'][None, ...],
            )[0]

        return sample

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Flip flags
        flip_flags_xyd = self.rng.binomial(1, 0.5, size=(2, 3)) != 0
        flip_flags_xyd = flip_flags_xyd & np.array([[self.hflip, self.vflip, self.hvflip]])

        sample0['image_arr'] = self.flip_image(sample0['image_arr'], flip_flags_xyd[0])
        sample1['image_arr'] = self.flip_image(sample1['image_arr'], flip_flags_xyd[1])

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = self.flip_image(sample0['mask_arr'], flip_flags_xyd[0])
            sample1['mask_arr'] = self.flip_image(sample1['mask_arr'], flip_flags_xyd[1])

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = self.flip_image(sample0['labels_arr'], flip_flags_xyd[0])
            sample1['labels_arr'] = self.flip_image(sample1['labels_arr'], flip_flags_xyd[1])

        if 'xf_cv' in sample0:
            # False -> 1, True -> -1
            flip_scale_xy = flip_flags_xyd[:, :2] * -2 + 1
            # Negative scale factors need to be combined with a translation whose value is (image_size - 1)
            # Mask the translation with the flip flags to only apply it where flipping is done
            flip_xlat_xy = flip_flags_xyd[:, :2] * (np.array([sample0['image_arr'].shape[:2][::-1],
                                                              sample1['image_arr'].shape[:2][::-1]]).astype(float) - 1)

            hv_flip_xf = affine.identity_xf(2)
            hv_flip_xf[flip_flags_xyd[:, 2]] = hv_flip_xf[flip_flags_xyd[:, 2], ::-1, :]

            xf01 = np.stack([sample0['xf_cv'], sample1['xf_cv']], axis=0)
            xf01 = affine.cat_nx2x3(
                hv_flip_xf,
                affine.translation_matrices(flip_xlat_xy),
                affine.scale_matrices(flip_scale_xy),
                xf01,
            )
            sample0['xf_cv'] = xf01[0]
            sample1['xf_cv'] = xf01[1]

        return (sample0, sample1)


class SegCVTransformTVT (SegTransform):
    """Apply a torchvision transform

    tvt_xform - the torchvision transform to apply
    apply_single - apply to single samples
    apply_pair0 - when transforming a pair of samples, apply to sample0
    apply_pair1 - when transforming a pair of samples, apply to sample1
    """
    def __init__(self, transform, apply_single=False, apply_pair0=False, apply_pair1=True):
        self.tvt_xform = transform
        self.apply_single = apply_single
        self.apply_pair0 = apply_pair0
        self.apply_pair1 = apply_pair1

    def _apply_to_image_array(self, img_arr):
        if img_arr.shape[2] == 4:
            alpha_channel = img_arr[:, :, 3:4]
        else:
            alpha_channel = None
        img_pil = Image.fromarray(img_arr[:, :, :3])
        img_pil = self.tvt_xform(img_pil)
        img_arr_rgb = np.array(img_pil)
        if alpha_channel is not None:
            return np.append(img_arr_rgb, alpha_channel, axis=2)
        else:
            return img_arr_rgb

    def transform_single(self, sample):
        if self.apply_single:
            sample = sample.copy()
            sample['image_arr'] = self._apply_to_image_array(sample['image_arr'])

        return sample

    def transform_pair(self, sample0, sample1):
        if self.apply_pair0:
            sample0 = sample0.copy()
            sample0['image_arr'] = self._apply_to_image_array(sample0['image_arr'])

        if self.apply_pair1:
            sample1 = sample1.copy()
            sample1['image_arr'] = self._apply_to_image_array(sample1['image_arr'])

        return (sample0, sample1)


class SegCVTransformNormalizeToTensor (SegTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform_single(self, sample):
        sample = sample.copy()

        # Convert image to float
        image = img_as_float(sample['image_arr'])

        if image.shape[2] == 4:
            # Has alpha channel introduced by padding
            # Split the image into RGB/alpha
            alpha_channel = image[:, :, 3:4]
            image = image[:, :, :3]

            # Account for the alpha during standardisation
            if self.mean is not None and self.std is not None:
                image = (image - (self.mean[None, None, :] * alpha_channel)) / self.std[None, None, :]
        else:
            # Standardisation
            if self.mean is not None and self.std is not None:
                image = (image - self.mean[None, None, :]) / self.std[None, None, :]

        # Convert to NCHW tensors
        assert image.shape[2] == 3
        sample['image'] = image.transpose(2, 0, 1).astype(np.float32)
        del sample['image_arr']
        if 'labels_arr' in sample:
            sample['labels'] = sample['labels_arr'][None, ...].astype(np.int64)
            del sample['labels_arr']
        if 'mask_arr' in sample:
            sample['mask'] = img_as_float(sample['mask_arr'])[None, ...].astype(np.float32)
            del sample['mask_arr']

        return sample

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Convert image to float
        image0 = img_as_float(sample0['image_arr'])
        image1 = img_as_float(sample1['image_arr'])

        if image0.shape[2] == 4:
            # Has alpha channel introduced by padding
            # Split the image into RGB/alpha
            alpha_channel0 = image0[:, :, 3:4]
            image0 = image0[:, :, :3]
            alpha_channel1 = image1[:, :, 3:4]
            image1 = image1[:, :, :3]

            # Account for the alpha during standardisation
            if self.mean is not None and self.std is not None:
                image0 = (image0 - (self.mean[None, None, :] * alpha_channel0)) / self.std[None, None, :]
                image1 = (image1 - (self.mean[None, None, :] * alpha_channel1)) / self.std[None, None, :]
        else:
            # Standardisation
            if self.mean is not None and self.std is not None:
                image0 = (image0 - self.mean[None, None, :]) / self.std[None, None, :]
                image1 = (image1 - self.mean[None, None, :]) / self.std[None, None, :]

        # Convert to NCHW tensors
        if image0.shape[2] != 3:
            raise ValueError('image0 should have 3 channels, not {}'.format(image0.shape[2]))
        if image1.shape[2] != 3:
            raise ValueError('image1 should have 3 channels, not {}'.format(image1.shape[2]))
        assert image1.shape[2] == 3
        sample0['image'] = image0.transpose(2, 0, 1).astype(np.float32)
        sample1['image'] = image1.transpose(2, 0, 1).astype(np.float32)
        del sample0['image_arr']
        del sample1['image_arr']
        if 'mask_arr' in sample0:
            sample0['mask'] = img_as_float(sample0['mask_arr'])[None, ...].astype(np.float32)
            sample1['mask'] = img_as_float(sample1['mask_arr'])[None, ...].astype(np.float32)
            del sample0['mask_arr']
            del sample1['mask_arr']

        if 'labels_arr' in sample0:
            sample0['labels'] = sample0['labels_arr'][None, ...].astype(np.int64)
            sample1['labels'] = sample1['labels_arr'][None, ...].astype(np.int64)
            del sample0['labels_arr']
            del sample1['labels_arr']

        return (sample0, sample1)