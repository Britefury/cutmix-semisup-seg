import pickle
import numpy as np
from datapipe import pascal_voc_dataset
from datapipe import camvid_dataset
from datapipe import cityscapes_dataset
from datapipe import isic2017_dataset
from datapipe import seg_transforms_cv
import torch.utils.data


def load_dataset(dataset, n_val, val_seed, n_sup, n_unsup, split_seed, split_path):
    val_rng = np.random.RandomState(val_seed)

    if split_path is not None:
        trainval_perm = pickle.load(open(split_path, 'rb'))
    else:
        trainval_perm = None

    if dataset == 'pascal':
        ds_src = pascal_voc_dataset.PascalVOCDataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx
    elif dataset == 'pascal_aug':
        ds_src = pascal_voc_dataset.PascalVOCDataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm, augmented=True)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx
    elif dataset == 'camvid':
        ds_src = camvid_dataset.CamVidDataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx
    elif dataset == 'cityscapes':
        ds_src = cityscapes_dataset.CityscapesDataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx
    elif dataset == 'isic2017':
        ds_src = isic2017_dataset.ISIC2017DataSource(n_val=n_val, val_rng=val_rng, trainval_perm=trainval_perm)
        ds_tgt = ds_src
        val_ndx_tgt = val_ndx_src = ds_src.val_ndx
        test_ndx_tgt = ds_src.test_ndx
    else:
        raise ValueError('Unknown dataset {}'.format(dataset))

    # Get training and validation sample indices
    split_rng = np.random.RandomState(split_seed)

    if split_path is not None:
        # The supplied split will have been used to shuffle the training samples, so
        # set train_perm to be the identity
        train_perm = np.arange(len(ds_src.train_ndx))
    else:
        # Random order
        train_perm = split_rng.permutation(len(ds_src.train_ndx))

    if ds_tgt is ds_src:
        if n_sup != -1:
            sup_ndx = ds_src.train_ndx[train_perm[:n_sup]]
            if n_unsup != -1:
                unsup_ndx = ds_src.train_ndx[train_perm[n_sup:n_sup + n_unsup]]
            else:
                unsup_ndx = ds_src.train_ndx[train_perm]
        else:
            sup_ndx = ds_src.train_ndx
            if n_unsup != -1:
                unsup_ndx = ds_src.train_ndx[train_perm[:n_unsup]]
            else:
                unsup_ndx = ds_src.train_ndx
    else:
        if n_sup != -1:
            sup_ndx = ds_src.train_ndx[train_perm[:n_sup]]
        else:
            sup_ndx = ds_src.train_ndx
        if n_unsup != -1:
            unsup_perm = split_rng.permutation(len(ds_tgt.train_ndx))
            unsup_ndx = ds_tgt.train_ndx[unsup_perm[:n_unsup]]
        else:
            unsup_ndx = ds_tgt.train_ndx

    return dict(
        ds_src=ds_src, ds_tgt=ds_tgt,
        val_ndx_tgt=val_ndx_tgt, val_ndx_src=val_ndx_src, test_ndx_tgt=test_ndx_tgt,
        sup_ndx=sup_ndx, unsup_ndx=unsup_ndx,
    )


def eval_data_pipeline(ds_src, ds_tgt, src_val_ndx, tgt_val_ndx, test_ndx,
                       batch_size, collate_fn, mean, std, num_workers):
    eval_transform = seg_transforms_cv.SegCVTransformNormalizeToTensor(mean, std)

    if ds_src is not ds_tgt:
        src_eval_ds = ds_src.dataset(labels=True, mask=False, xf=False,
                                     transforms=eval_transform,
                                     pipeline_type='cv')
        src_val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(src_eval_ds, src_val_ndx),
                                                     batch_size, collate_fn=collate_fn,
                                                     num_workers=num_workers)
    else:
        src_val_loader = None

    tgt_eval_ds = ds_tgt.dataset(labels=True, mask=False, xf=False,
                                 transforms=eval_transform,
                                 pipeline_type='cv', include_indices=True)
    tgt_val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(tgt_eval_ds, tgt_val_ndx),
                                                 batch_size, collate_fn=collate_fn,
                                                 num_workers=num_workers)

    if test_ndx is not None:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(tgt_eval_ds, test_ndx),
                                                  batch_size, collate_fn=collate_fn,
                                                  num_workers=num_workers)
    else:
        test_loader = None

    return src_val_loader, tgt_val_loader, test_loader
