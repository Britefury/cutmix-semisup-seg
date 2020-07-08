import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.metrics import confusion_matrix


def fast_cm(tru, pred, num_classes):
    """
    Compute confusion matrix quickly using `np.bincount`
    :param tru: true class
    :param pred: predicted class
    :param num_classes: number of classes
    :return: confusion matrix
    """
    bin = tru * num_classes + pred
    h = np.bincount(bin, minlength=num_classes*num_classes)
    return h.reshape((num_classes, num_classes))

def per_class_i_and_u_cm(pred, tru, num_classes, ignore_value=None):
    intersection = []
    union = []

    mask = tru != ignore_value

    for cls_i in range(num_classes):
        if ignore_value is None:
            p_mask = pred == cls_i
            t_mask = tru == cls_i
        else:
            p_mask = (pred == cls_i) & mask
            t_mask = (tru == cls_i) & mask

        intersection.append((p_mask & t_mask).sum())
        union.append((p_mask | t_mask).sum())

    cm = fast_cm(tru[mask], pred[mask], num_classes)

    return np.array(intersection), np.array(union), cm



class EvaluatorIoU (object):
    def __init__(self, num_classes, fill_holes=False):
        if fill_holes:
            if num_classes != 2:
                raise ValueError('num_classes must be 2 if fill_holes is True')
        self.num_classes = num_classes
        self.fill_holes = fill_holes
        self.intersection = np.zeros((num_classes,))
        self.union = np.zeros((num_classes,))
        self.cm = np.zeros((num_classes, num_classes))

    def sample(self, truth, prediction, ignore_value=None):
        if self.fill_holes:
            pred_bin = binary_fill_holes(prediction != 0)
            prediction = pred_bin.astype(int)
        i, u, cm = per_class_i_and_u_cm(prediction, truth, self.num_classes, ignore_value=ignore_value)
        self.intersection += i
        self.union += u
        self.cm += cm

    def score(self):
        return self.intersection.astype(float) / np.maximum(self.union.astype(float), 1.0)

