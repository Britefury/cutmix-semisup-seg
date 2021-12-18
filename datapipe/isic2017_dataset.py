import os
import pickle
import settings
import numpy as np
from PIL import Image
from datapipe import seg_data


class ISIC2017Accessor (seg_data.SegAccessor):
    def __len__(self):
        return len(self.ds.x_names)

    def get_image_pil(self, sample_i):
        return self.ds.get_pil_image(self.ds.x_names[sample_i])

    def get_labels_arr(self, sample_i):
        pil_img = self.ds.get_pil_image(self.ds.y_names[sample_i])
        y = (np.array(pil_img) >= 127).astype(np.int32)
        return y


def _get_isic2017_path(exists=False):
    return settings.get_data_path(
        config_name='isic2017',
        dnnlib_template=os.path.join('<DATASETS>', 'research', 'gfrench', 'isic2017',
                                     'isic2017_segmentation.zip'),
        exists=exists
    )


class ISIC2017DataSource (seg_data.ZipDataSource):
    def __init__(self, n_val, val_rng, trainval_perm):
        super(ISIC2017DataSource, self).__init__(_get_isic2017_path(exists=True))

        sample_names = set()

        for filename in self.zip_file.namelist():
            x_name, ext = os.path.splitext(filename)
            if x_name.endswith('_x') and ext.lower() == '.png':
                sample_name = x_name[:-2]
                sample_names.add(sample_name)

        sample_names = list(sample_names)
        sample_names.sort()

        self.x_names = ['{}_x.png'.format(name) for name in sample_names]
        self.y_names = ['{}_y.png'.format(name) for name in sample_names]
        self.sample_names = sample_names

        self.train_ndx = np.array([i for i in range(len(self.sample_names))
                                   if self.sample_names[i].startswith('train/')])
        self.val_ndx = np.array([i for i in range(len(self.sample_names))
                                 if self.sample_names[i].startswith('val/')])
        self.test_ndx = None

        if n_val > 0:
            # We want a hold out validation set: use validation set as test
            # and split the training set
            self.test_ndx = self.val_ndx

            if trainval_perm is not None:
                assert len(trainval_perm) == len(self.train_ndx)
                trainval = self.train_ndx[trainval_perm]
            else:
                trainval = self.train_ndx[val_rng.permutation(len(self.train_ndx))]
            self.train_ndx = trainval[:-n_val]
            self.val_ndx = trainval[-n_val:]
        else:
            # Use trainval_perm to re-order the training samples
            if trainval_perm is not None:
                assert len(trainval_perm) == len(self.train_ndx)
                self.train_ndx = self.train_ndx[trainval_perm]

        self.class_names = ['background', 'lesion']

        self.num_classes = len(self.class_names)

        mean_std = pickle.loads(self._read_file_from_zip_as_bytes('rgb_mean_std.pkl'))
        self.rgb_mean = mean_std['rgb_mean']
        self.rgb_std = mean_std['rgb_std']


    def dataset(self, labels, mask, xf, transforms=None, pipeline_type='cv', include_indices=False):
        return ISIC2017Accessor(self, labels, mask, xf, transforms=transforms, pipeline_type=pipeline_type,
                                include_indices=include_indices)


    def get_mean_std(self):
        # For now:
        return (self.rgb_mean, self.rgb_std)
