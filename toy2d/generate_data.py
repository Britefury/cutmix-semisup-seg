import click
import pickle
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_erosion
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.datasets import make_s_curve
from skimage.util import img_as_float
from skimage.color import rgb2grey, hsv2rgb
from skimage.transform import downscale_local_mean
from skimage.filters import roberts
import cv2
from batchup import data_source
import torch.utils.data


def blend(a, b, t):
    return a + (b - a) * t

class Dataset2D (object):
    def __init__(self, X, y, img_size):
        self.img_size = img_size
        self.img_scale = np.array(img_size).astype(float)

        self.X = X
        self.y = y

        px_grid_x, px_grid_y = np.meshgrid(np.arange(self.img_size[1]), np.arange(self.img_size[0]))
        self.px_grid = np.stack([px_grid_y, px_grid_x], axis=2) + 0.5

    def load_supervised(self, path):
        raise NotImplementedError('Abstract for {}'.format(type(self)))

    def img_to_real(self, x):
        return (x / self.img_scale) * 2.0 - 1.0

    def real_to_img(self, x):
        return (x + 1.0) * 0.5 * self.img_scale


class ClassificationDataset2D (Dataset2D):
    def __init__(self, X, y, img_size, sup_indices, unsup_indices):
        super(ClassificationDataset2D, self).__init__(X, y, img_size)

        self.sup_X = self.X[sup_indices]
        self.sup_y = self.y[sup_indices]
        self.unsup_X = self.X[unsup_indices]
        self.unsup_y = self.y[unsup_indices]

        self.sup_X_img = self.real_to_img(self.sup_X)
        self.unsup_X_img = self.real_to_img(self.unsup_X)

        X_img = self.real_to_img(X)

        # Generate sample density image
        density_img_bins = np.arange(self.img_size[0]*16) / 16.0
        dens_img, _, _ = np.histogram2d(X_img[:, 0], X_img[:, 1], bins=(density_img_bins, density_img_bins))
        dens_img = dens_img.astype(float)
        dens_img = downscale_local_mean(dens_img, (16,16)) * 256.0
        self.dens_img = 1.0 - (0.75 ** dens_img)

        self.px_grid_vis = self.img_to_real(self.px_grid.reshape((-1, 2)))
        # self.px_grid_ds = data_source.ArrayDataSource([self.px_grid_vis])



    def load_supervised(self, path):
        with open(path, 'rb') as f_in:
            data = pickle.load(f_in)
            self.sup_X = data['clf_sup_X']
            self.sup_y = data['clf_sup_y']
            self.sup_X_img = self.real_to_img(self.sup_X)


    def semisup_image_plot(self, pred_y1, pred_grad):
        vis = np.zeros(self.img_size + (3,), dtype=float)
        vis += 1.0 - self.dens_img[:, :, None]
        if pred_y1.ndim == 2:
            pred_y1 = pred_y1.reshape(self.img_size)
        vis = blend(vis, np.array([[[0.0, 0.75, 0.0]]]), pred_y1[:, :, None] * 0.3)

        if pred_grad is not None:
            if pred_grad.ndim == 2:
                pred_grad = pred_grad.reshape(self.img_size)
            pred_grad = pred_grad / max(abs(pred_grad).max(), 1e-30)
            pred_grad = np.sqrt(pred_grad)
            vis = blend(vis, np.array([[[0.0, 0.0, 1.0]]]), pred_grad[:, :, None] * 0.5)

        vis = np.clip(vis, 0.0, 1.0)

        vis = (vis * 255.0).astype(np.uint8)

        for i in np.where(self.sup_y == 0)[0]:
            cv2.circle(vis, (int(self.sup_X_img[i,1]), int(self.sup_X_img[i,0])), 5, (255, 128, 0), 2)
        for i in np.where(self.sup_y == 1)[0]:
            cv2.circle(vis, (int(self.sup_X_img[i,1]), int(self.sup_X_img[i,0])), 5, (0, 0, 255), 2)

        return vis


class SplitClassificationDataset2D (ClassificationDataset2D):
    def __init__(self, X, y, img_size, n_sup, balance_classes, rng):
        if balance_classes:
            n_classes = y.max()+1
            sup_indices = []
            unsup_indices = []
            n_per_cls = n_sup // n_classes
            for cls_i in range(n_classes):
                cls_ndx = np.arange(len(y))[y == cls_i]
                rng.shuffle(cls_ndx)
                sup_indices.append(cls_ndx[:n_per_cls])
                unsup_indices.append(cls_ndx)
            sup_indices = np.concatenate(sup_indices, axis=0)
            unsup_indices = np.concatenate(unsup_indices, axis=0)
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_sup, random_state=rng)
            _, sup_indices = next(splitter.split(y, y))
            unsup_indices = np.arange(len(y))


        super(SplitClassificationDataset2D, self).__init__(X, y, img_size, sup_indices, unsup_indices)



class ClassificationDatasetFromImage2D (SplitClassificationDataset2D):
    def __init__(self, image, X, y, img_size, n_sup, balance_classes, rng):
        self.img_size = img_size
        self.img_scale = np.array(img_size).astype(float)

        super(ClassificationDatasetFromImage2D, self).__init__(X, y, img_size, n_sup, balance_classes, rng)

        self.image = image
        self.image_edges = roberts(self.image)


    def img_to_real(self, x):
        return (x / self.img_scale) * 2.0 - 1.0

    def real_to_img(self, x):
        return (x + 1.0) * 0.5 * self.img_scale


    def semisup_image_plot(self, pred_y1, pred_grad):
        vis = np.zeros(self.img_size + (3,), dtype=float)
        vis += 1.0 - self.dens_img[:, :, None]
        if pred_y1.ndim < 2:
            pred_y1 = pred_y1.reshape(self.img_size)
        vis = blend(vis, np.array([[[0.0, 0.75, 0.0]]]), pred_y1[:, :, None] * 0.3)

        if pred_grad is not None:
            if pred_grad.ndim == 2:
                pred_grad = pred_grad.reshape(self.img_size)
            pred_grad = pred_grad / max(abs(pred_grad).max(), 1e-30)
            pred_grad = np.sqrt(pred_grad)
            vis = blend(vis, np.array([[[0.0, 0.0, 1.0]]]), pred_grad[:, :, None] * 0.5)

        vis = blend(vis, np.array([[[1.0, 0.0, 1.0]]]), self.image_edges[:, :, None] * 0.5)

        vis = np.clip(vis, 0.0, 1.0)

        vis = (vis * 255.0).astype(np.uint8)

        for i in np.where(self.sup_y == 0)[0]:
            cv2.circle(vis, (int(self.sup_X_img[i,1]), int(self.sup_X_img[i,0])), 5, (255, 128, 0), 2)
        for i in np.where(self.sup_y == 1)[0]:
            cv2.circle(vis, (int(self.sup_X_img[i,1]), int(self.sup_X_img[i,0])), 5, (0, 0, 255), 2)

        return vis


def classification_dataset_from_image(image_path: str, region_erode_radius: int, img_noise_std: float, n_sup: int,
                                      balance_classes:bool, rng: np.random.RandomState) -> ClassificationDatasetFromImage2D:
    img = np.array(Image.open(image_path))
    img = img_as_float(rgb2grey(img))
    img_bin = img >= 0.5

    img_size = img_bin.shape

    if region_erode_radius > 0:
        img_cls_1 = binary_erosion(img_bin, iterations=region_erode_radius)
        img_cls_0 = binary_erosion(~img_bin, iterations=region_erode_radius)
    else:
        img_cls_1 = img_bin
        img_cls_0 = ~img_bin

    samples_0_y, samples_0_x = np.where(img_cls_0)
    samples_1_y, samples_1_x = np.where(img_cls_1)

    X_img_0 = np.stack([samples_0_y, samples_0_x], axis=1)
    X_img_1 = np.stack([samples_1_y, samples_1_x], axis=1)
    y_0 = np.zeros((len(X_img_0),), dtype=int)
    y_1 = np.ones((len(X_img_1),), dtype=int)
    X_img = np.append(X_img_0, X_img_1, axis=0)
    y = np.append(y_0, y_1, axis=0)

    X_img = X_img + rng.normal(loc=0, scale=img_noise_std, size=X_img.shape)

    X_real = ((X_img) / np.array(img_size)) * 2 - 1

    return ClassificationDatasetFromImage2D(img, X_real, y, img_size, n_sup, balance_classes, rng)


def spiral_classification_dataset(n_sup: int, balance_classes:bool, rng: np.random.RandomState, N: int=5000,
                                  spiral_radius: float=20, img_size=(256, 256)) -> ClassificationDataset2D:
    # Generate spiral dataset
    # Taking the sqrt of the randomly drawn radii ensures uniform sample distribution
    # Using plain uniform distribution results in samples concentrated at the centre
    radius0 = np.sqrt(rng.uniform(low=1.0, high=spiral_radius ** 2, size=(N,)))
    radius1 = np.sqrt(rng.uniform(low=1.0, high=spiral_radius ** 2, size=(N,)))
    theta0 = radius0 * 0.5
    theta1 = radius1 * 0.5 + np.pi
    radius = np.append(radius0, radius1, axis=0)
    theta = np.append(theta0, theta1, axis=0)
    X = np.stack([np.sin(theta) * radius, np.cos(theta) * radius], axis=1)
    y = np.append(np.zeros(radius0.shape, dtype=int), np.ones(radius1.shape, dtype=int), axis=0)

    X = X + rng.normal(size=X.shape) * 0.2

    X = X / spiral_radius

    return SplitClassificationDataset2D(X, y, img_size, n_sup, balance_classes, rng)


def crosshatch_classification_dataset(rng: np.random.RandomState, grid_size: int, points_per_cell: int,
                                      cell_off_std: float=0.05, n_sup:int=2, img_size=(256, 256)) -> \
        ClassificationDataset2D:
    # Generate cross-hatch dataset
    cell_size = 2.0 / grid_size
    cell_off_std = cell_off_std * cell_size

    g = np.linspace(-1, 1, grid_size+1)
    x0, y0 = np.meshgrid(g, g)
    X0 = np.stack([y0, x0], axis=2).reshape((-1, 2))
    X0 = np.repeat(X0, points_per_cell, axis=0)

    x1, y1 = np.meshgrid(g[:-1]+cell_size*0.5, g[:-1]+cell_size*0.5)
    X1 = np.stack([y1, x1], axis=2).reshape((-1, 2))
    X1 = np.repeat(X1, points_per_cell, axis=0)

    X = np.append(X0, X1, axis=0)
    X = X + rng.normal(size=X.shape) * cell_off_std
    y = np.append(np.zeros((len(X0),), dtype=int), np.ones((len(X1),), dtype=int), axis=0)

    sup_X = np.array([[0.0, 0.0], [cell_size*0.5, cell_size*0.5]])
    sup_y = np.array([0, 1])

    if n_sup == -1:
        sup_indices = np.arange(len(y))
        unsup_indices = np.arange(2) + len(y)
    else:
        unsup_indices = np.arange(len(y))
        sup_indices = np.arange(2) + len(y)

    X = np.append(X, sup_X, axis=0)
    y = np.append(y, sup_y, axis=0)

    ds = ClassificationDataset2D(X, y, img_size, sup_indices, unsup_indices)

    ds.cell_size = cell_size
    ds.cell_off_std = cell_off_std

    return ds




@click.group()
def cli():
    pass

@cli.command('clf')
@click.argument('out_path', type=click.Path(file_okay=True, writable=True))
@click.option('--image_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option('--region_erode_radius', type=int, default=35)
@click.option('--noise_std', type=float, default=6.0)
@click.option('--n_sup', type=int, default=10)
@click.option('--balance_split', is_flag=True, default=False)
@click.option('--seed', type=int, default=12345)
def generate_classification(out_path, image_path, region_erode_radius, noise_std, n_sup, balance_split, seed):
    import pickle

    rng = np.random.RandomState(seed)

    if image_path is not None:
        ds = classification_dataset_from_image(image_path, region_erode_radius, noise_std, n_sup, balance_split, rng)
    else:
        ds = spiral_classification_dataset(n_sup, balance_split, rng)

    data = dict(clf_sup_X=ds.sup_X, clf_unsup_X=ds.unsup_X, clf_sup_y=ds.sup_y, clf_unsup_y=ds.unsup_y)

    with open(out_path, 'wb') as f_out:
        pickle.dump(data, f_out)

if __name__ == '__main__':
    cli()