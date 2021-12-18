"""
Pascal VOC dataset

Place VOC2012 dataset in 'VOC2012' directory.
For training, you will need the augmented labels. Download http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip.
The folder structure should be like:
VOC2012/JPEGImages
       /SegmentationClassAug
"""
import os, pickle
import tqdm
import numpy as np
import settings
from PIL import Image
from datapipe import seg_data


class PascalVOCAccessor (seg_data.SegAccessor):
    def __len__(self):
        return len(self.ds.sample_names)

    def get_image_pil(self, sample_i):
        return self.ds._get_input_pil(sample_i)

    def get_labels_arr(self, sample_i):
        img = self.ds._get_unmapped_labels_arr(sample_i)
        if self.ds.class_map is not None:
            img = self.ds.class_map[img]
        return img


def _load_names(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line != '']


def _get_pascal_path(exists=False):
    return settings.get_data_path(
        config_name='pascal_voc',
        dnnlib_template=os.path.join(
            '<DATASETS>', 'research', 'pascal_voc'),
        exists=exists
    )

class PascalVOCDataSource (seg_data.DataSource):
    def __init__(self, n_val, val_rng, trainval_perm, fg_class_subset=None, augmented=False):
        pascal_path = _get_pascal_path(exists=True)
        self.class_map = None

        if augmented:
            train_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'train_aug.txt')
            val_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'val.txt')

            train_names = _load_names(train_aug_names_path)
            val_names = _load_names(val_aug_names_path)

            self.sample_names = list(set(train_names + val_names))
            self.sample_names.sort()

            name_to_index = {name: name_i for name_i, name in enumerate(self.sample_names)}
            self.train_ndx = np.array([name_to_index[name] for name in train_names])
            self.val_ndx = np.array([name_to_index[name] for name in val_names])

            self.semantic_y_paths = [os.path.join(pascal_path, 'SegmentationClassAug', '{}.png'.format(name)) for name in self.sample_names]

        else:
            train_names_path = os.path.join(pascal_path, 'ImageSets', 'Segmentation', 'train.txt')
            val_names_path = os.path.join(pascal_path, 'ImageSets', 'Segmentation', 'val.txt')

            train_names = _load_names(train_names_path)
            val_names = _load_names(val_names_path)

            self.sample_names = list(set(train_names + val_names))
            self.sample_names.sort()

            name_to_index = {name: name_i for name_i, name in enumerate(self.sample_names)}
            self.train_ndx = np.array([name_to_index[name] for name in _load_names(train_names_path)])
            self.val_ndx = np.array([name_to_index[name] for name in _load_names(val_names_path)])

            self.semantic_y_paths = [os.path.join(pascal_path, 'SegmentationClass', '{}.png'.format(name)) for name in self.sample_names]

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

        self.x_paths = [os.path.join(pascal_path, 'JPEGImages', '{}.jpg'.format(name)) for name in self.sample_names]

        self.num_classes = 21

        if fg_class_subset is not None:
            fg_subset_str = '-'.join([str(x) for x in fg_class_subset])
            valid_image_indices_path = os.path.join(pascal_path, 'valid_images_fg_subset_{}.pkl'.format(fg_subset_str))
            if os.path.exists(valid_image_indices_path):
                with open(valid_image_indices_path, 'rb') as f_valid_indices:
                    valid_images = pickle.load(f_valid_indices)
            else:
                fg_class_set = set(fg_class_subset)
                valid_images = []
                for sample_i in tqdm.tqdm(range(len(self.sample_names))):
                    y = np.array(self._get_unmapped_labels_arr(sample_i))
                    classes_present = set(np.unique(y.flatten()))
                    if len(classes_present.intersection(fg_class_set)) > 0:
                        valid_images.append(sample_i)
                valid_images = np.array(valid_images)

                with open(valid_image_indices_path, 'wb') as f_valid_indices:
                    pickle.dump(valid_images, f_valid_indices)

            self.num_classes = len(fg_class_subset) + 1

            valid_images_set = set(valid_images)
            self.train_ndx = np.array([i for i in self.train_ndx if i in valid_images_set])
            self.val_ndx = np.array([i for i in self.val_ndx if i in valid_images_set])

            self.class_map = np.zeros((256,), dtype=int)
            fg_class_subset = np.array(fg_class_subset)
            self.class_map[fg_class_subset] = np.arange(len(fg_class_subset)) + 1
            self.class_map = self.class_map.astype(np.uint8)
            self.class_map[255] = 255

            self.class_weights = np.append(self.class_weights[:1], self.class_weights[np.array(fg_class_subset)], axis=0)


    def _get_input_pil(self, sample_i):
        path = self.x_paths[sample_i]
        img = Image.open(path)
        img.load()
        return img

    def _get_unmapped_labels_arr(self, sample_i):
        path = self.semantic_y_paths[sample_i]
        img = Image.open(path)
        img.load()
        return np.array(img)


    def dataset(self, labels, mask, xf, transforms=None, pipeline_type='cv', include_indices=False):
        return PascalVOCAccessor(self, labels, mask, xf, transforms=transforms, pipeline_type=pipeline_type,
                                 include_indices=include_indices)

    def get_mean_std(self):
        # For now:
        return np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
