import os
import settings
import numpy as np
from datapipe import seg_data

CLASS_NAMES_WITH_VOID = [
    'unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground',

    'road', 'sidewalk', 'parking', 'rail_track',

    'building', 'wall', 'fence', 'guard_rail', 'bridge', 'tunnel',

    'pole', 'pole_group', 'traffic_light', 'traffic_sign',

    'vegetation', 'terrain', 'sky',

    'person', 'rider',

    'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',

    'license_plate'
]

VOID_CLASS_NAMES = [
    'unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground',

    'parking', 'rail_track',

    'guard_rail', 'bridge', 'tunnel',

    'pole_group',

    'caravan', 'trailer',

    'license_plate'
]

VOID_CLASS_INDICES = np.array([CLASS_NAMES_WITH_VOID.index(name) for name in VOID_CLASS_NAMES])

CLASS_NAMES = [name for name in CLASS_NAMES_WITH_VOID if name not in VOID_CLASS_NAMES]


class CityscapesAccessor (seg_data.SegAccessor):
    def __len__(self):
        return len(self.ds.x_names)

    def get_image_pil(self, sample_i):
        return self.ds.get_pil_image(self.ds.x_names[sample_i])

    def get_labels_arr(self, sample_i):
        pil_img = self.ds.get_pil_image(self.ds.y_names[sample_i])
        y = np.array(pil_img)
        if not self.ds.with_void:
            y = self.ds.non_void_mapping[y]
        return y


def _get_cityscapes_path(exists=False):
    return settings.get_data_path(
        config_name='cityscapes',
        dnnlib_template=os.path.join('<DATASETS>', 'research', 'gfrench', 'cityscapes_full',
                                     'cityscapes_segmentation.zip'),
        exists=exists
    )


class CityscapesDataSource (seg_data.ZipDataSource):
    def __init__(self, n_val, val_rng, trainval_perm, with_void=False):
        super(CityscapesDataSource, self).__init__(_get_cityscapes_path(exists=True))

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



        self.class_names_with_void = CLASS_NAMES_WITH_VOID
        self.void_class_names = VOID_CLASS_NAMES
        self.void_class_indices = VOID_CLASS_INDICES
        self.class_names = CLASS_NAMES

        self.with_void = with_void

        self.non_void_mapping = []
        out_cls_i = 0
        for cls_i, name in enumerate(self.class_names_with_void):
            if name in self.void_class_names:
                self.non_void_mapping.append(255)
            else:
                self.non_void_mapping.append(out_cls_i)
                out_cls_i += 1
        self.non_void_mapping = np.array(self.non_void_mapping)

        self.num_classes_with_void = len(self.class_names_with_void)
        self.num_classes = len(self.class_names)


    def dataset(self, labels, mask, xf, transforms=None, pipeline_type='cv', include_indices=False):
        return CityscapesAccessor(self, labels, mask, xf, transforms=transforms, pipeline_type=pipeline_type,
                                  include_indices=include_indices)


    def get_mean_std(self):
        # For now:
        return (np.array([0.485, 0.456, 0.406]),
                np.array([0.229, 0.224, 0.225]))
