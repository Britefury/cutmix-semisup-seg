import click


@click.command()
@click.argument('leftimg8bit_trainvaltest_zip_path', type=click.Path(readable=True))
@click.argument('gtfine_trainvaltest_zip_path', type=click.Path(readable=True))
@click.option('--downsample', type=int, default=2)
def convert(leftimg8bit_trainvaltest_zip_path, gtfine_trainvaltest_zip_path, downsample):
    import os
    import tqdm
    import numpy as np
    from PIL import Image
    from skimage.transform import downscale_local_mean
    from datapipe import cityscapes_dataset
    import zipfile

    def downsample_label_img(y, downsample):
        n_classes = y.max() + 1
        y_one_hot = (y[:, :, None] == np.arange(n_classes)[None, None, :]).astype(int)
        y_one_hot = y_one_hot.reshape(
            (y_one_hot.shape[0]//downsample, downsample, y_one_hot.shape[1]//downsample, downsample, n_classes)
        )
        y_one_hot = y_one_hot.sum(axis=(1, 3))
        return np.argmax(y_one_hot, axis=2)

    out_path = cityscapes_dataset._get_cityscapes_path(exists=False)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    print('Writing data to {}'.format(out_path))
    x_zip = zipfile.ZipFile(leftimg8bit_trainvaltest_zip_path, 'r')
    y_zip = zipfile.ZipFile(gtfine_trainvaltest_zip_path, 'r')
    out_zip = zipfile.ZipFile(out_path, 'w')
    names_to_process = [name for name in x_zip.namelist()
                        if os.path.splitext(name)[1].lower() == '.png' and not name.startswith('leftImg8bit/test')]
    for name in tqdm.tqdm(names_to_process):
        left_8bit = name
        sample_name = os.path.splitext(name)[0].replace('_leftImg8bit', '').replace('leftImg8bit/', '')
        gt_fine_semantic_name = 'gtFine/{}_gtFine_labelIds.png'.format(sample_name)

        x_img = np.array(Image.open(x_zip.open(left_8bit, 'r')))
        y_img = np.array(Image.open(y_zip.open(gt_fine_semantic_name, 'r')))

        if downsample != 1:
            x_img = downscale_local_mean(x_img, (downsample, downsample, 1)).astype(np.uint8)
            y_img = downsample_label_img(y_img, downsample)

        x_name = '{}_x.png'.format(sample_name)
        y_name = '{}_y.png'.format(sample_name)

        Image.fromarray(x_img).save(out_zip.open(x_name, 'w'), 'PNG')
        Image.fromarray(y_img.astype(np.uint32)).save(out_zip.open(y_name, 'w'), 'PNG')



if __name__ == '__main__':
    convert()