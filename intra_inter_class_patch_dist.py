import click

@click.command()
@click.argument('out_path', type=click.Path(writable=True))
@click.option('--dataset', type=click.Choice(['camvid', 'cityscapes', 'pascal', 'pascal_aug', 'gtav',
                                              'inria_aerial', 'isic2017']), default='cityscapes')
@click.option('--patch_size', type=int, default=225)
@click.option('--n_patches', type=int, default=1000)
@click.option('--n_neighbours', type=int, default=1000)
@click.option('--batch_size', type=int, default=-1)
@click.option('--batch', type=int, default=0)
@click.option('--show_progress', is_flag=True, default=False)
@click.option('--batch_index_one_based', is_flag=True, default=False)
@click.option('--load_choice', type=click.Path(readable=True, exists=True))
@click.option('--save_choice', type=click.Path(writable=True))
@click.option('--seed', type=int, default=12345)
def intra_inter_class_patch_dist(out_path, dataset, patch_size, n_patches, n_neighbours,
                                 batch_size, batch, show_progress, batch_index_one_based,
                                 load_choice, save_choice, seed):
    import pickle
    import tqdm
    import sys
    import numpy as np
    from skimage.util import img_as_float

    from datapipe import datasets
    import patch_dist

    if batch_index_one_based:
        batch -= 1

    print('Command line:')
    print(' '.join(sys.argv))

    print('Loading dataset...', flush=True)
    ds = datasets.load_dataset(dataset, n_val=0, val_seed=0,
                               n_sup=-1, n_unsup=-1, split_seed=12345, split_path=None)['ds_src']

    if show_progress:
        progress_fn = tqdm.tqdm
    else:
        progress_fn = lambda x, *args, **kwargs: x

    rng = np.random.RandomState(seed)

    def choose_anchors_and_negatives(xs, ys, sample_indices, n_patches, patch_shape, rng):
        """
        Choose anchor and negative patch locations

        Anchor patches are centred on pixels adjacent to a class boundary
        Negative patches neighbour the anchor patches on the other side of the class boundary

        :param xs: source of input images
        :param ys: source of ground truth label images
        :param sample_indices: indices of training samples to use
        :param n_patches: number of patch pairs to choose
        :param patch_shape: patch size as `(H, W)` tuple
        :param rng: random number generator
        :return: index array of shape `(N, [img_i, dir_i, y, x, cls])` where:
            img_i: image index
            dir_i: direction index [0=left, 1=right, 2=north, 3=south]
            y: patch centre Y
            x: patch centre X
            cls: ground truth class of central pixel
        """
        patch_shape = np.array(patch_shape)
        pad = (patch_shape - 1) // 2
        border = pad + 1

        img_dir_y_x_cls = []
        for img_i in progress_fn(sample_indices):
            y = ys[int(img_i)]
            y = np.array(y)
            # neigh_cls_chg is a list of 4 2D boolean arrays that indicate for
            # each pixel if the neighbour to the left, right, above or below
            # (respectively) has a different class.
            neigh_cls_chg = patch_dist.neighbouring_pixels_class_change(y)

            # dir_ijy: an `(N, [dir, i, j, y])` (Nx4) array that has an entry
            # for each pixel in the image whose index is `img_i`
            # whose neighbour has a different class, where `i` and `j`
            # identify the row and column of the pixel, `y` is the
            # class of the pixel and `dir` indicates the direction to
            # the different neighbour:
            # 0=left, 1=right, 2=above, 3=below

            dir_ijy = []
            for dir_i, chg_map in enumerate(neigh_cls_chg):
                i, j = np.where(chg_map)
                i_valid = (i > border[0]) & (i < (y.shape[0] - border[0]))
                j_valid = (j > border[1]) & (j < (y.shape[1] - border[1]))
                i = i[i_valid & j_valid]
                j = j[i_valid & j_valid]

                img_dir_y_x_cls.append(np.stack([np.ones_like(i) * img_i, np.ones_like(i) * dir_i,
                                                 i, j, y[i, j]], axis=1))

        img_dir_y_x_cls = np.concatenate(img_dir_y_x_cls, axis=0)

        # Randomly choose `n_patches` neighbouring pixel pairs
        choice = rng.permutation(len(img_dir_y_x_cls))[:n_patches]
        img_dir_y_x_cls = img_dir_y_x_cls[choice]

        return img_dir_y_x_cls


    def extract_anchor_and_negative_patches(xs, ys, img_dir_y_x_cls, patch_shape):
        neighbour_offsets = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])

        anchor_patches = []
        negative_patches = []

        for row in progress_fn(img_dir_y_x_cls):
            q_ij = row[2:4]
            q_neigh_ij = q_ij + neighbour_offsets[row[1]]

            # Check that the class is as expected
            img_y = np.array(ys[int(row[0])])
            assert img_y[q_ij[0], q_ij[1]] == row[4]
            assert img_y[q_neigh_ij[0], q_neigh_ij[1]] != img_y[q_ij[0], q_ij[1]]

            x = img_as_float(xs[int(row[0])])
            q_patch = patch_dist.extract_patch(x, patch_shape, q_ij)
            q_neigh_patch = patch_dist.extract_patch(x, patch_shape, q_neigh_ij)

            anchor_patches.append(q_patch)
            negative_patches.append(q_neigh_patch)

        anchor_patches = np.stack(anchor_patches, axis=0)
        negative_patches = np.stack(negative_patches, axis=0)

        # Return:
        # img_dir_ijy: `(N, [img_index, dir, i, j, y])` array indicating the pixel pairs chosen
        # patches: `(N, patch_shape[0], patch_shape[1], channel)` shaped array that is the
        #       image content of the patches centred on the chosen pixels
        # neighbour_patches: `(N, patch_shape[0], patch_shape[1], channel)` shaped array that
        #       is the image content of the neighbouring patches
        return anchor_patches, negative_patches


    if load_choice is not None:
        print('Loading choice of anchor and negative patches from {}'.format(load_choice))
        with open(load_choice, 'rb') as f_in:
            anchor_negative_ids = pickle.load(f_in)
    else:
        print('Choosing anchor and negative patches...', flush=True)
        anchor_negative_ids = choose_anchors_and_negatives(ds.x, ds.semantic_y, ds.train_ndx, n_patches,
                                                           (patch_size, patch_size), rng)

        if save_choice is not None:
            print('Saving choice of anchor and negative patches to {}'.format(save_choice))
            with open(save_choice, 'wb') as f_out:
                pickle.dump(anchor_negative_ids, f_out)


    # Select batch we are working on
    if batch_size == -1:
        batch_size = len(anchor_negative_ids)

    patch_i0 = batch * batch_size
    patch_i1 = (batch + 1) * batch_size
    batch_anchor_negative_ids = anchor_negative_ids[patch_i0:patch_i1]

    print('Extracting anchor and negative patches...', flush=True)
    batch_anchor_patches, batch_negative_patches = extract_anchor_and_negative_patches(#
        ds.x, ds.semantic_y, batch_anchor_negative_ids, (patch_size, patch_size))


    def class_distances(ys, img_dir_y_x_cls, anchor_patches, n_neighbours):
        n_patches = len(anchor_patches)

        same_image_intra_class_dists = [None for _ in range(n_patches)]
        same_image_intra_class_coords = [None for _ in range(n_patches)]
        same_image_inter_class_dists = [None for _ in range(n_patches)]
        same_image_inter_class_coords = [None for _ in range(n_patches)]

        other_image_intra_class_dists = [np.zeros((0,)) for _ in range(n_patches)]
        other_image_intra_class_coords = [np.zeros((0, 3), dtype=int) for _ in range(n_patches)]
        other_image_inter_class_dists = [np.zeros((0,)) for _ in range(n_patches)]
        other_image_inter_class_coords = [np.zeros((0, 3), dtype=int) for _ in range(n_patches)]

        for img_i in progress_fn(ds.train_ndx):
            image = img_as_float(ds.x[int(img_i)])
            y = np.array(ys[int(img_i)])

            for patch_i, dist_map in enumerate(patch_dist.sliding_window_distance_to_patches_generator(
                    image, anchor_patches)):
                # Timing this loop indicates that the majority (~90%) of the time is spent computing
                # the distance map. For Cityscapes, 0.25s per distance map, 0.03s for argsort, 0.01s for the rest

                img_dir_y_x_cls_row = img_dir_y_x_cls[patch_i]

                # Get a mask that identifies the pixels that belong to the same class
                # as the query pixel
                intra_class_mask = y == img_dir_y_x_cls_row[4]
                inter_class_mask = (y != img_dir_y_x_cls_row[4]) & (y != 255)

                dist_map_flat = dist_map.flatten()
                intra_class_mask_flat = intra_class_mask.flatten()
                inter_class_mask_flat = inter_class_mask.flatten()
                order = np.argsort(dist_map_flat)
                # Filter order so that we only retain elements corresponding to pixel that are of
                # the same class as the query pixel
                intra_class_order = order[intra_class_mask_flat[order]]
                inter_class_order = order[inter_class_mask_flat[order]]
                assert intra_class_mask_flat[intra_class_order].all()
                assert inter_class_mask_flat[inter_class_order].all()

                # order = order[:n_neighbours]
                intra_class_order = intra_class_order[:n_neighbours]
                inter_class_order = inter_class_order[:n_neighbours]

                # dists = dist_map_flat[order]
                # coords = np.unravel_index(order, dist_map.shape)
                # coords = np.stack(coords, axis=1)
                # coords = np.concatenate([np.ones((len(coords), 1), dtype=int) * img_i,
                #                          coords], axis=1)

                intra_class_dists = dist_map_flat[intra_class_order]
                intra_class_coords = np.unravel_index(intra_class_order, dist_map.shape)
                intra_class_coords = np.stack(intra_class_coords, axis=1)
                intra_class_coords = np.concatenate([np.ones((len(intra_class_coords), 1), dtype=int) * img_i,
                                                     intra_class_coords], axis=1)

                inter_class_dists = dist_map_flat[inter_class_order]
                inter_class_coords = np.unravel_index(inter_class_order, dist_map.shape)
                inter_class_coords = np.stack(inter_class_coords, axis=1)
                inter_class_coords = np.concatenate([np.ones((len(inter_class_coords), 1), dtype=int) * img_i,
                                                     inter_class_coords], axis=1)

                if img_i == img_dir_y_x_cls_row[0]:
                    # same_image_dists[patch_i] = dists
                    # same_image_coords[patch_i] = coords
                    same_image_intra_class_dists[patch_i] = intra_class_dists
                    same_image_intra_class_coords[patch_i] = intra_class_coords
                    same_image_inter_class_dists[patch_i] = inter_class_dists
                    same_image_inter_class_coords[patch_i] = inter_class_coords
                else:
                    # d = other_image_dists[patch_i]
                    # c = other_image_coords[patch_i]
                    # d = np.append(d, dists, axis=0)
                    # c = np.append(c, coords, axis=0)
                    # order = np.argsort(d)[:n_neighbours]
                    # d = d[order]
                    # c = c[order]
                    # other_image_dists[patch_i] = d
                    # other_image_coords[patch_i] = c

                    d = other_image_intra_class_dists[patch_i]
                    c = other_image_intra_class_coords[patch_i]
                    d = np.append(d, intra_class_dists, axis=0)
                    c = np.append(c, intra_class_coords, axis=0)
                    order = np.argsort(d)[:n_neighbours]
                    d = d[order]
                    c = c[order]
                    other_image_intra_class_dists[patch_i] = d
                    other_image_intra_class_coords[patch_i] = c

                    d = other_image_inter_class_dists[patch_i]
                    c = other_image_inter_class_coords[patch_i]
                    d = np.append(d, inter_class_dists, axis=0)
                    c = np.append(c, inter_class_coords, axis=0)
                    order = np.argsort(d)[:n_neighbours]
                    d = d[order]
                    c = c[order]
                    other_image_inter_class_dists[patch_i] = d
                    other_image_inter_class_coords[patch_i] = c

        return dict(
            same_image_intra_class_dists=same_image_intra_class_dists,
            same_image_intra_class_coords=same_image_intra_class_coords,
            same_image_inter_class_dists=same_image_inter_class_dists,
            same_image_inter_class_coords=same_image_inter_class_coords,
            other_image_intra_class_dists=other_image_intra_class_dists,
            other_image_intra_class_coords=other_image_intra_class_coords,
            other_image_inter_class_dists=other_image_inter_class_dists,
            other_image_inter_class_coords=other_image_inter_class_coords,
        )


    anchor_negative_dist = np.sqrt(((batch_anchor_patches - batch_negative_patches) ** 2).sum(axis=(1, 2, 3)))

    print('Computing distances...', flush=True)
    results = class_distances(ds.semantic_y, batch_anchor_negative_ids, batch_anchor_patches, n_neighbours)

    results['anchor_negative_img_dir_y_x_cls'] = batch_anchor_negative_ids
    results['boundary_dists'] = anchor_negative_dist

    with open(out_path, 'wb') as f_out:
        pickle.dump(results, f_out)




if __name__ == '__main__':
    intra_inter_class_patch_dist()