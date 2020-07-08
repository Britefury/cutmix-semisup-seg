import click


@click.command()
def convert():
    import os
    import urllib.request
    from datapipe import pascal_voc_dataset

    _AUG_TRAIN_LIST_URL = r'http://raw.githubusercontent.com/hfslyc/AdvSemiSeg/master/dataset/voc_list/train_aug.txt'
    _AUG_VAL_LIST_URL = r'http://raw.githubusercontent.com/hfslyc/AdvSemiSeg/master/dataset/voc_list/val.txt'

    pascal_path = pascal_voc_dataset._get_pascal_path(exists=False)
    seg_aug_dir_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug')
    train_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'train_aug.txt')
    val_aug_names_path = os.path.join(pascal_path, 'ImageSets', 'SegmentationAug', 'val.txt')

    # Download the names for the augmented training set if its not there
    if not os.path.exists(train_aug_names_path):
        os.makedirs(seg_aug_dir_path, exist_ok=True)
        urllib.request.urlretrieve(_AUG_TRAIN_LIST_URL, train_aug_names_path)
        assert os.path.exists(train_aug_names_path)

    # Download the names for the augmented validation set if its not there
    if not os.path.exists(val_aug_names_path):
        os.makedirs(seg_aug_dir_path, exist_ok=True)
        urllib.request.urlretrieve(_AUG_VAL_LIST_URL, val_aug_names_path)
        assert os.path.exists(val_aug_names_path)


if __name__ == '__main__':
    convert()
