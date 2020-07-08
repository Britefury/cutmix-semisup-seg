import click


@click.command()
@click.argument('isic_zips_dir', type=click.Path(readable=True))
@click.option('--out_size', type=str, default='248,248')
def convert_isic(isic_zips_dir, out_size):
    import os
    import tqdm
    import pickle
    import numpy as np
    from PIL import Image
    from skimage.util import img_as_float
    import cv2
    from datapipe import isic2017_dataset

    import zipfile

    if ',' in out_size:
        h, w = out_size.split(',')
        h = int(h.strip())
        w = int(w.strip())
        out_size = h, w
    else:
        out_size = out_size.strip()
        if len(out_size) > 0:
            out_size = int(out_size.strip())
        else:
            out_size = None

    def process_zip_pair(out_zip, out_folder, in_x_zip, in_y_zip, x_folder, y_folder):
        x_paths_to_process = []
        rgb_sum = np.zeros((3,))
        rgb2_sum = np.zeros((3,))
        rgb_n = 0
        for x_path in in_x_zip.namelist():
            name, ext = os.path.splitext(x_path)
            if ext.lower() == '.jpg' and not name.lower().endswith('_superpixels'):
                x_paths_to_process.append(x_path)
        for x_path in tqdm.tqdm(x_paths_to_process):
            x_dir, x_filename = os.path.split(x_path)
            x_name, x_ext = os.path.splitext(x_filename)
            y_path = '{}/{}_segmentation.png'.format(y_folder, x_name)

            out_x_filename = '{}/{}_x.png'.format(out_folder, x_name)
            out_y_filename = '{}/{}_y.png'.format(out_folder, x_name)

            x_img = np.array(Image.open(in_x_zip.open(x_path, 'r')))
            y_img = np.array(Image.open(in_y_zip.open(y_path, 'r')))

            if out_size is None:
                pass
            elif isinstance(out_size, int):
                min_size = min(x_img.shape[0], x_img.shape[1])
                scale_factor = float(out_size) / float(min_size)
                x_img = cv2.resize(x_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                y_img = cv2.resize(y_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            elif isinstance(out_size, tuple):
                x_img = cv2.resize(x_img, out_size[::-1], interpolation=cv2.INTER_AREA)
                y_img = cv2.resize(y_img, out_size[::-1], interpolation=cv2.INTER_AREA)
            else:
                raise RuntimeError

            Image.fromarray(x_img).save(out_zip.open(out_x_filename, 'w'), 'PNG')
            Image.fromarray(y_img).save(out_zip.open(out_y_filename, 'w'), 'PNG')

            rgb = img_as_float(x_img)
            rgb_sum += rgb.sum(axis=(0,1))
            rgb2_sum += (rgb**2).sum(axis=(0,1))
            rgb_n += rgb.shape[0] * rgb.shape[1]

        rgb_mean = rgb_sum / rgb_n
        rgb_std = np.sqrt(rgb2_sum/rgb_n - rgb_sum*rgb_sum/rgb_n/rgb_n)

        return rgb_mean, rgb_std

    train_x_zip_path = os.path.join(isic_zips_dir, 'ISIC-2017_Training_Data.zip')
    train_y_zip_path = os.path.join(isic_zips_dir, 'ISIC-2017_Training_Part1_GroundTruth.zip')
    val_x_zip_path = os.path.join(isic_zips_dir, 'ISIC-2017_Validation_Data.zip')
    val_y_zip_path = os.path.join(isic_zips_dir, 'ISIC-2017_Validation_Part1_GroundTruth.zip')

    out_path = isic2017_dataset._get_isic2017_path(exists=False)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    print('Writing data to {}'.format(out_path))
    train_x_zip = zipfile.ZipFile(train_x_zip_path, 'r')
    train_y_zip = zipfile.ZipFile(train_y_zip_path, 'r')
    val_x_zip = zipfile.ZipFile(val_x_zip_path, 'r')
    val_y_zip = zipfile.ZipFile(val_y_zip_path, 'r')
    out_zip = zipfile.ZipFile(out_path, 'w')

    print('Processing training set...')
    rgb_mean, rgb_std = process_zip_pair(
        out_zip, 'train', train_x_zip, train_y_zip, 'ISIC-2017_Training_Data', 'ISIC-2017_Training_Part1_GroundTruth')
    print('Processing validation set...')
    process_zip_pair(out_zip, 'val', val_x_zip, val_y_zip, 'ISIC-2017_Validation_Data', 'ISIC-2017_Validation_Part1_GroundTruth')

    print('Writing mean and std-dev...')
    with out_zip.open('rgb_mean_std.pkl', 'w') as f_mean_std:
        mean_std = dict(rgb_mean=rgb_mean, rgb_std=rgb_std)
        pickle.dump(mean_std, f_mean_std)




if __name__ == '__main__':
    convert_isic()