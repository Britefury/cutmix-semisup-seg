import io
import os
import math
import zipfile
import threading
import itertools
from collections import namedtuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from datapipe import affine


class SegAccessor (Dataset):
    def __init__(self, ds, labels, mask, xf, transforms, pipeline_type='cv', include_indices=False):
        """
        Generates samples.

        Can generate samples for either a Pillow (pipeline_type='pil') or OpenCV (pipeline_type='cv')
        based pipeline.

        Pillow samples take the form:
            {'image_pil': PIL.Image,                # input image
             [optional] 'labels_pil': PIL.Image,    # labels image
             [optional] 'mask_pil': PIL.Image,      # mask image
             [optional] 'xf_pil': np.array}         # transformation as NumPy array

        OpenCV samples take the form:
            {'image_arr': np.array,                 # input image as a `(H, W, C)` array
             [optional] 'labels_arr': np.array,     # labels image as a `(H, W)` array
             [optional] 'mask_arr': np.array,       # mask image as a `(H, W)` array
             [optional] 'xf_cv': np.array}          # transformation as NumPy array

        :param ds: data source to load from
        :param labels: flag indicating if the ground truth labels should be loaded
        :param mask: flag indicating if mask should be loaded
        :param xf: flag indicating if transformation should be loaded
        :param transforms: optional transformation to apply to each sample when retrieved
        :param pipeline_type: pipeline type 'pil' | 'cv'
        :param include_indices: if True, include sample index in each sample
        """
        super(SegAccessor, self).__init__()

        if pipeline_type not in {'pil', 'cv'}:
            raise ValueError('pipeline_type should be either \'pil\' or \'cv\', not {}'.format(pipeline_type))

        self.ds = ds
        self.labels_flag = labels
        self.mask_flag = mask
        self.xf_flag = xf
        self.transforms = transforms
        self.pipeline_type = pipeline_type
        self.include_indices = include_indices

    def __len__(self):
        raise NotImplementedError('Abstract')

    def get_image_pil(self, sample_i):
        raise NotImplementedError('Abstract')

    def get_labels_arr(self, sample_i):
        raise NotImplementedError('Abstract')

    def __getitem__(self, sample_i):
        sample = {}

        image = self.get_image_pil(sample_i)
        size_xy = image.size
        sample['image_size_yx'] = np.array(size_xy[::-1])
        if self.pipeline_type == 'pil':
            sample['image_pil'] = image
        elif self.pipeline_type == 'cv':
            sample['image_arr'] = np.array(image)
        else:
            raise RuntimeError

        if self.labels_flag:
            labels = self.get_labels_arr(sample_i)
            if self.pipeline_type == 'pil':
                sample['labels_pil'] = Image.fromarray(labels)
            elif self.pipeline_type == 'cv':
                sample['labels_arr'] = labels.astype(np.int32)
            else:
                raise RuntimeError

        if self.mask_flag:
            if self.pipeline_type == 'pil':
                sample['mask_pil'] = Image.new('L', size_xy, 255)
            elif self.pipeline_type == 'cv':
                sample['mask_arr'] = np.full(size_xy[::-1], 255, dtype=np.uint8)
            else:
                raise RuntimeError

        if self.xf_flag:
            xf = affine.identity_xf(1)[0]
            if self.pipeline_type == 'pil':
                sample['xf_pil'] = xf
            elif self.pipeline_type == 'cv':
                sample['xf_cv'] = xf
            else:
                raise RuntimeError

        if self.include_indices:
            sample['index'] = int(sample_i)

        if self.transforms is not None:
            sample = self.transforms.apply(sample)
        return sample


def save_prediction(out_dir, pred_y_arr, sample_name):
    path = os.path.join(out_dir, "{}.png".format(sample_name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(pred_y_arr.astype(np.uint32)).save(path)


class DataSource (object):
    def save_prediction_by_index(self, out_dir, pred_y_arr, sample_index):
        save_prediction(out_dir, pred_y_arr, self.sample_names[sample_index])

    def get_mean_std(self):
        # For now:
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class _ZipFileWrapper(object):
    """
    A wrapper for a ZipFile object that can be pickled

    PyTorch data loaders use multi-processing and pickling to send objects between processes.
    This wrapper can be pickled between processes, as it only pickles the path and
    will open the zip file on demand as necessary.
    """

    def __init__(self, path):
        self.path = path
        self.__zip_file = None
        self.__pid = None

    def __getstate__(self):
        return {'path': self.path}

    def __setstate__(self, state):
        self.path = state['path']
        self.__zip_file = None

    def get(self):
        my_pid = os.getpid()
        if self.__zip_file is None or my_pid != self.__pid:
            self.__zip_file = zipfile.ZipFile(self.path, 'r')
            self.__pid = my_pid
        return self.__zip_file


class ZipDataSource(DataSource):
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.__zip_wrapper = _ZipFileWrapper(zip_path)

    @property
    def zip_file(self):
        return self.__zip_wrapper.get()

    def _read_file_from_zip_as_bytes(self, name):
        with self.zip_file.open(name) as f:
            return f.read()

    def get_pil_image(self, name):
        img_str = self._read_file_from_zip_as_bytes(name)
        img = Image.open(io.BytesIO(img_str))
        img.load()
        return img


class SegCollate (object):
    def __init__(self, block_size, batch_aug_fn=None):
        self.block_size = block_size
        self.batch_aug_fn = batch_aug_fn

    @staticmethod
    def _compute_padding(in_size, size):
        if in_size != size:
            dh = size[0] - in_size[0]
            dw = size[1] - in_size[1]
            dh0 = dh // 2
            dh1 = dh - dh0
            dw0 = dw // 2
            dw1 = dw - dw0
            return [[0, 0], [dh0, dh1], [dw0, dw1]]
        else:
            return None

    @staticmethod
    def _apply_padding_to_tensor(t, padding, value=0):
        return np.pad(t, padding, mode='constant', constant_values=value)

    @staticmethod
    def _pad_sample(sample, size):
        padding = SegCollate._compute_padding(sample['image'].shape[1:3], size)

        if padding is not None:
            sample = sample.copy()
            sample['image'] = SegCollate._apply_padding_to_tensor(sample['image'], padding, value=0)
            if 'labels' in sample:
                sample['labels'] = SegCollate._apply_padding_to_tensor(sample['labels'], padding, value=255)
            if 'mask' in sample:
                sample['mask'] = SegCollate._apply_padding_to_tensor(sample['mask'], padding, value=255)
            if 'xf_pil' in sample:
                dy, dx = padding[1][0], padding[2][0]
                sample['xf_pil'] = affine.cat_nx2x3(sample['xf_pil'][None, ...], affine.translation_matrices(np.array([[dx, dy]])))[0]
            if 'xf_cv' in sample:
                dy, dx = padding[1][0], padding[2][0]
                sample['xf_cv'] = affine.cat_nx2x3(affine.translation_matrices(np.array([[dx, dy]])), sample['xf_cv'][None, ...])[0]

        return sample

    @staticmethod
    def _compute_xf_0_to_1(pair):
        sample0 = pair['sample0']
        sample1 = pair['sample1']
        if 'xf_cv' in sample0 and 'xf_cv' in sample1:
            xf0_to_1_cv = affine.cat_nx2x3(sample1['xf_cv'][None, ...], affine.inv_nx2x3(sample0['xf_cv'][None, ...]))
            xf0_to_1 = affine.cv_to_torch(xf0_to_1_cv, sample1['image'].shape[1:3])
            pair['xf0_to_1_cv'] = xf0_to_1_cv[0]
            pair['xf0_to_1'] = xf0_to_1[0].astype(np.float32)
        elif 'xf_pil' in sample0 and 'xf_pil' in sample1:
            xf0_to_1_pil = affine.cat_nx2x3(affine.inv_nx2x3(sample0['xf_pil'][None, ...]), sample1['xf_pil'][None, ...])
            xf0_to_1 = affine.pil_to_torch(xf0_to_1_pil, sample1['image'].shape[1:3])
            pair['xf0_to_1_pil'] = xf0_to_1_pil[0]
            pair['xf0_to_1'] = xf0_to_1[0].astype(np.float32)
        return pair

    @staticmethod
    def _convert_xf(sample):
        if 'xf_pil' in sample:
            sample['xf'] = affine.pil_to_torch(sample['xf_pil'][None, ...], sample['image'].shape[1:3],
                                               sample['image_size_yx'])[0].astype(np.float32)
            del sample['xf_pil']
        elif 'xf_cv' in sample:
            sample['xf'] = affine.cv_to_torch(sample['xf_cv'][None, ...], sample['image'].shape[1:3],
                                              sample['image_size_yx'])[0].astype(np.float32)
            del sample['xf_cv']
        return sample

    def __call__(self, batch):
        is_paired = 'sample0' in batch[0]

        # Compute maximum image size
        size = (0, 0)
        if is_paired:
            # Paired
            for pair in batch:
                size = max(size[0], pair['sample0']['image'].shape[1]), max(size[1], pair['sample0']['image'].shape[2])
                size = max(size[0], pair['sample1']['image'].shape[1]), max(size[1], pair['sample1']['image'].shape[2])
        else:
            for sample in batch:
                size = max(size[0], sample['image'].shape[1]), max(size[1], sample['image'].shape[2])

        # Round size up to block size
        rounded_size = (round(math.ceil(size[0] / self.block_size[0]) * self.block_size[0]),
                        round(math.ceil(size[1] / self.block_size[1]) * self.block_size[1]))

        if is_paired:
            for pair in batch:
                pair['sample0'] = SegCollate._pad_sample(pair['sample0'], rounded_size)
                pair['sample1'] = SegCollate._pad_sample(pair['sample1'], rounded_size)
                SegCollate._compute_xf_0_to_1(pair)
                pair['sample0'] = SegCollate._convert_xf(pair['sample0'])
                pair['sample1'] = SegCollate._convert_xf(pair['sample1'])
        else:
            batch = [SegCollate._pad_sample(sample, rounded_size) for sample in batch]
            batch = [SegCollate._convert_xf(sample) for sample in batch]

        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)

        return default_collate(batch)


class RepeatSampler(Sampler):
    r"""Repeated sampler

    Arguments:
        data_source (Dataset): dataset to sample from
        sampler (Sampler): sampler to draw from repeatedly
        repeats (int): number of repetitions or -1 for infinite
    """

    def __init__(self, sampler, repeats=-1):
        if repeats < 1 and repeats != -1:
            raise ValueError('repeats should be positive or -1')
        self.sampler = sampler
        self.repeats = repeats

    def __iter__(self):
        if self.repeats == -1:
            reps = itertools.repeat(self.sampler)
            return itertools.chain.from_iterable(reps)
        else:
            reps = itertools.repeat(self.sampler, self.repeats)
            return itertools.chain.from_iterable(reps)

    def __len__(self):
        if self.repeats == -1:
            return 2 ** 62
        else:
            return len(self.sampler) * self.repeats
