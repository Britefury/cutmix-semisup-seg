import job_helper
import click

@job_helper.job('toy2d_train', enumerate_job_names=False)
def train_toy2d(submit_config: job_helper.SubmitConfig, dataset, region_erode_radius, img_noise_std,
                n_sup, balance_classes, seed,
                sup_path, model, n_hidden, hidden_size, hidden_act, norm_layer,
                perturb_noise_std, dist_contour_range,
                conf_thresh, conf_avg,
                cons_weight, cons_loss_fn, cons_no_dropout,
                learning_rate, teacher_alpha,
                num_epochs, batch_size, render_cons_grad, render_pred, device,
                save_output):
    settings = locals().copy()
    del settings['submit_config']

    import sys

    print('Command line:')
    print(' '.join(sys.argv))
    print('Settings:')
    print(', '.join(['{}={}'.format(k, settings[k]) for k in sorted(settings.keys())]))

    import os
    import numpy as np
    import time
    import cv2
    from scipy.ndimage.morphology import distance_transform_edt
    import optim_weight_ema
    from toy2d import generate_data
    from datapipe.seg_data import RepeatSampler

    import torch, torch.nn as nn, torch.nn.functional as F
    import torch.utils.data

    rng = np.random.RandomState(seed)

    # Generate/load the dataset
    if dataset.startswith('img:'):
        # Generate a dataset from a black and white image
        image_path = dataset[4:]
        ds = generate_data.classification_dataset_from_image(image_path, region_erode_radius, img_noise_std, n_sup, balance_classes, rng)
        image = ds.image
    elif dataset == 'spiral':
        # Generate a spiral dataset
        ds = generate_data.spiral_classification_dataset(n_sup, balance_classes, rng)
        image = None
    else:
        print('Unknown dataset {}, should be spiral or img:<path>'.format(dataset))
        return

    # If a path to a supervised dataset has been provided, load it
    if sup_path is not None:
        ds.load_supervised(sup_path)

    # If we are constraining perturbations to lie along the contours of the distance map to the ground truth class boundary
    if dist_contour_range > 0.0:
        if image is None:
            print('Constraining perturbations to lying on distance map contours is only supported for \'image\' experiments')
            return
        img_1 = image >= 0.5
        # Compute signed distance map to boundary
        dist_1 = distance_transform_edt(img_1)
        dist_0 = distance_transform_edt(~img_1)
        dist_map = dist_1 * img_1 + -dist_0 * (~img_1)
    else:
        dist_map = None

    # PyTorch device
    torch_device = torch.device(device)

    # Convert perturbation noise std-dev to [y,x]
    try:
        perturb_noise_std = np.array([float(x.strip()) for x in perturb_noise_std.split(',')])
    except ValueError:
        perturb_noise_std = np.array([6.0, 6.0])

    # Assume that perturbation noise std-dev is in pixel space (for image experiments), so convert
    perturb_noise_std_real_scale = perturb_noise_std / ds.img_scale * 2.0
    perturb_noise_std_real_scale = torch.tensor(perturb_noise_std_real_scale, dtype=torch.float, device=torch_device)

    # Define the neural network model (an MLP)
    class Network (nn.Module):
        def __init__(self):
            super(Network, self).__init__()

            self.drop = nn.Dropout()

            hidden = []
            chn_in = 2
            for i in range(n_hidden):
                if norm_layer == 'spectral_norm':
                    hidden.append(nn.utils.spectral_norm(nn.Linear(chn_in, hidden_size)))
                elif norm_layer == 'weight_norm':
                    hidden.append(nn.utils.weight_norm(nn.Linear(chn_in, hidden_size)))
                else:
                    hidden.append(nn.Linear(chn_in, hidden_size))

                if norm_layer == 'batch_norm':
                    hidden.append(nn.BatchNorm1d(hidden_size))
                elif norm_layer == 'group_norm':
                    hidden.append(nn.GroupNorm(4, hidden_size))

                if hidden_act == 'relu':
                    hidden.append(nn.ReLU())
                elif hidden_act == 'lrelu':
                    hidden.append(nn.LeakyReLU(0.01))
                else:
                    raise ValueError

                chn_in = hidden_size
            self.hidden = nn.Sequential(*hidden)

            # Final layer; 2-class output
            self.l_final = nn.Linear(chn_in, 2)

        def forward(self, x, use_dropout=True):
            x = self.hidden(x)
            if use_dropout:
                x = self.drop(x)
            x = self.l_final(x)
            return x

    # Build student network, optimizer and supervised loss criterion
    student_net = Network().to(torch_device)
    student_params = list(student_net.parameters())

    student_optimizer = torch.optim.Adam(student_params, lr=learning_rate)
    classification_criterion = nn.CrossEntropyLoss()

    # Build teacher network and optimizer
    if model == 'mean_teacher':
        teacher_net = Network().to(torch_device)
        teacher_params = list(teacher_net.parameters())
        for param in teacher_params:
            param.requires_grad = False
        teacher_optimizer = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, ema_alpha=teacher_alpha)
        pred_net = teacher_net
    else:
        teacher_net = None
        teacher_optimizer = None
        pred_net = student_net

    # Robust BCE helper
    def robust_binary_crossentropy(pred, tgt):
        inv_tgt = -tgt + 1.0
        inv_pred = -pred + 1.0 + 1e-6
        return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))

    # If we are constraining perturbations to lie on distance map contours, load the distance map as a Torch tensor
    if dist_contour_range > 0.0:
        t_dist_map = torch.tensor(dist_map[None, None, ...], dtype=torch.float, device=torch_device)
    else:
        t_dist_map = None

    # Helper function to compute confidence thresholding factor
    def conf_factor(teacher_pred_prob):
        # Compute confidence
        conf_tea = torch.max(teacher_pred_prob, 1)[0]
        conf_tea = conf_tea.detach()
        # Compute factor based on threshold and `conf_avg` flag
        if conf_thresh > 0.0:
            conf_fac = (conf_tea >= conf_thresh).float()
        else:
            conf_fac = torch.ones(conf_tea.shape, dtype=torch.float, device=conf_tea.device)
        if conf_avg:
            conf_fac = torch.ones_like(conf_fac) * conf_fac.mean()
        return conf_fac

    # Helper function that constrains consistency loss to operate only when perturbations lie along
    # distance map contours.
    # When this feature is enabled, it masks to zero the loss for any unsupervised sample whose random perturbation
    # deviates too far from the distance map contour
    def dist_map_weighting(t_dist_map, batch_u_X, batch_u_X_1):
        if t_dist_map is not None and dist_contour_range > 0:
            # For each sample in `batch_u_X` and `batch_u_X_1`, both of which are
            # of shape `[n_points, [y,x]]` we want to get the value from the
            # distance map. For this we use `torch.nn.functional.grid_sample`.
            # This function expects grid look-up co-ordinates to have
            # the shape `[batch, height, width, [x, y]]`.
            # We reshape `batch_u_X` and `batch_u_X_1` to `[1, 1, n_points, [x,y]]` and stack along
            # the height dimension, making two rows to send to `grid_sample`.
            # The final shape will be `[1, 2, n_points, [x,y]]`:
            # 1 sample (1 image)
            # 2 rows; batch_u_X and batch_u_X_1
            # n_points columns
            # (x,y)
            # `[n_points, [y,x]]` -> `[1, 1, n_points, [x,y]]`
            sample_points_0 = torch.cat([batch_u_X[:, 1].view(1, 1, -1, 1),
                                         batch_u_X[:, 0].view(1, 1, -1, 1)], dim=3)
            # `[n_points, [y,x]]` -> `[1, 1, n_points, [x,y]]`
            sample_points_1 = torch.cat([batch_u_X_1[:, 1].view(1, 1, -1, 1),
                                         batch_u_X_1[:, 0].view(1, 1, -1, 1)], dim=3)
            # -> `[1, 2, n_points, [x,y]]`
            sample_points = torch.cat([sample_points_0, sample_points_1], dim=1)
            # Get distance to class boundary from distance map
            dist_from_boundary = F.grid_sample(t_dist_map, sample_points)
            # Get the squared difference between the distances from `batch_u_X` to the boundary
            # and the distances from `batch_u_X_1` to the boundary.
            delta_dist_sqr = (dist_from_boundary[0, 0, 0, :] - dist_from_boundary[0, 0, 1, :]).pow(2)
            # Per-sample loss mask based on difference between distances
            weight = (delta_dist_sqr <= (dist_contour_range*dist_contour_range)).float()

            return weight
        else:
            return torch.ones(len(batch_u_X), dtype=torch.float, device=batch_u_X.device)

    # Supervised dataset, sampler and loader
    sup_dataset = torch.utils.data.TensorDataset(torch.tensor(ds.sup_X, dtype=torch.float),
                                                 torch.tensor(ds.sup_y, dtype=torch.long))
    sup_sampler = RepeatSampler(torch.utils.data.RandomSampler(sup_dataset))
    sup_sep_loader = torch.utils.data.DataLoader(sup_dataset, batch_size, sampler=sup_sampler, num_workers=1)

    # Unsupervised dataset, sampler and loader
    unsup_dataset = torch.utils.data.TensorDataset(torch.tensor(ds.unsup_X, dtype=torch.float))
    unsup_sampler = torch.utils.data.RandomSampler(unsup_dataset)
    unsup_loader = torch.utils.data.DataLoader(unsup_dataset, batch_size, sampler=unsup_sampler, num_workers=1)

    # Complete dataset and loader
    all_dataset = torch.utils.data.TensorDataset(torch.tensor(ds.X, dtype=torch.float))
    all_loader = torch.utils.data.DataLoader(all_dataset, 16384, shuffle=False, num_workers=1)

    # Grid points used to render visualizations
    vis_grid_dataset = torch.utils.data.TensorDataset(torch.tensor(ds.px_grid_vis, dtype=torch.float))
    vis_grid_loader = torch.utils.data.DataLoader(vis_grid_dataset, 16384, shuffle=False, num_workers=1)

    # Evaluation mode initially
    student_net.eval()
    if teacher_net is not None:
        teacher_net.eval()

    # Compute the magnitude of the gradient of the consistency loss at the logits
    def consistency_loss_logit_grad_mag(batch_u_X):
        u_shape = batch_u_X.shape

        batch_u_X_1 = batch_u_X + torch.randn(u_shape, dtype=torch.float, device=torch_device) * \
                                  perturb_noise_std_real_scale[None, :]

        student_optimizer.zero_grad()

        grads = [None]

        if teacher_net is not None:
            teacher_unsup_logits = teacher_net(batch_u_X).detach()
        else:
            teacher_unsup_logits = student_net(batch_u_X)
        teacher_unsup_prob = F.softmax(teacher_unsup_logits, dim=1)
        student_unsup_logits = student_net(batch_u_X_1)
        def grad_hook(grad):
            grads[0] = torch.sqrt((grad*grad).sum(dim=1))

        student_unsup_logits.register_hook(grad_hook)
        student_unsup_prob = F.softmax(student_unsup_logits, dim=1)

        weight = dist_map_weighting(t_dist_map, batch_u_X, batch_u_X_1)

        mod_fac = conf_factor(teacher_unsup_prob) * weight

        if cons_loss_fn == 'bce':
            aug_loss = robust_binary_crossentropy(student_unsup_prob, teacher_unsup_prob)
            aug_loss = aug_loss.mean(dim=1) * mod_fac
            unsup_loss = aug_loss.mean()
        elif cons_loss_fn == 'var':
            d_aug_loss = student_unsup_prob - teacher_unsup_prob
            aug_loss = d_aug_loss * d_aug_loss
            aug_loss = aug_loss.mean(dim=1) * mod_fac
            unsup_loss = aug_loss.mean()
        elif cons_loss_fn == 'logits_var':
            d_aug_loss = student_unsup_logits - teacher_unsup_logits
            aug_loss = d_aug_loss * d_aug_loss
            aug_loss = aug_loss.mean(dim=1) * mod_fac
            unsup_loss = aug_loss.mean()
        else:
            raise ValueError

        unsup_loss.backward()

        return (grads[0].cpu().numpy(),)

    # Helper function for rendering an output image for visualization
    def render_output_image():
        # Generate output for plotting
        with torch.no_grad():
            vis_pred = []
            vis_grad = [] if render_cons_grad else None
            for (batch_X,) in vis_grid_loader:
                batch_X = batch_X.to(torch_device)
                batch_pred_logits = pred_net(batch_X)
                if render_pred == 'prob':
                    batch_vis = F.softmax(batch_pred_logits, dim=1)[:, 1]
                elif render_pred == 'class':
                    batch_vis = torch.argmax(batch_pred_logits, dim=1)
                else:
                    raise ValueError('Unknown prediction render {}'.format(render_pred))
                batch_vis = batch_vis.detach().cpu().numpy()
                vis_pred.append(batch_vis)

                if render_cons_grad:
                    batch_grad = consistency_loss_logit_grad_mag(batch_X)
                    vis_grad.append(batch_grad.detach().cpu().numpy())

            vis_pred = np.concatenate(vis_pred, axis=0)
            if render_cons_grad:
                vis_grad = np.concatenate(vis_grad, axis=0)

        out_image = ds.semisup_image_plot(vis_pred, vis_grad)
        return out_image

    # Output image for first frame
    if save_output and submit_config.run_dir is not None:
        plot_path = os.path.join(submit_config.run_dir, 'epoch_{:05d}.png'.format(0))
        cv2.imwrite(plot_path, render_output_image())
    else:
        cv2.imshow('Vis', render_output_image())
        k = cv2.waitKey(1)


    # Train
    print('|sup|={}'.format(len(ds.sup_X)))
    print('|unsup|={}'.format(len(ds.unsup_X)))
    print('|all|={}'.format(len(ds.X)))
    print('Training...')

    terminated = False
    for epoch in range(num_epochs):
        t1 = time.time()
        student_net.train()
        if teacher_net is not None:
            teacher_net.train()

        batch_sup_loss_accum = 0.0
        batch_conf_mask_sum_accum = 0.0
        batch_cons_loss_accum = 0.0
        batch_N_accum = 0.0
        for sup_batch, unsup_batch in zip(sup_sep_loader, unsup_loader):
            (batch_X, batch_y) = sup_batch
            (batch_u_X,) = unsup_batch

            batch_X = batch_X.to(torch_device)
            batch_y = batch_y.to(torch_device)
            batch_u_X = batch_u_X.to(torch_device)

            # Apply perturbation to generate `batch_u_X_1`
            aug_perturbation = torch.randn(batch_u_X.shape, dtype=torch.float, device=torch_device)
            batch_u_X_1 = batch_u_X + aug_perturbation * perturb_noise_std_real_scale[None, :]

            # Supervised loss path
            student_optimizer.zero_grad()
            student_sup_logits = student_net(batch_X)
            sup_loss = classification_criterion(student_sup_logits, batch_y)

            if cons_weight > 0.0:
                # Consistency loss path

                # Logits are computed differently depending on model
                if model == 'mean_teacher':
                    teacher_unsup_logits = teacher_net(batch_u_X, use_dropout=not cons_no_dropout).detach()
                    student_unsup_logits = student_net(batch_u_X_1, use_dropout=not cons_no_dropout)
                elif model == 'pi':
                    teacher_unsup_logits = student_net(batch_u_X, use_dropout=not cons_no_dropout)
                    student_unsup_logits = student_net(batch_u_X_1, use_dropout=not cons_no_dropout)
                elif model == 'pi_onebatch':
                    batch_both = torch.cat([batch_u_X, batch_u_X_1], dim=0)
                    both_unsup_logits = student_net(batch_both, use_dropout=not cons_no_dropout)
                    teacher_unsup_logits = both_unsup_logits[:len(batch_u_X)]
                    student_unsup_logits = both_unsup_logits[len(batch_u_X):]
                else:
                    raise RuntimeError

                # Compute predicted probabilities
                teacher_unsup_prob = F.softmax(teacher_unsup_logits, dim=1)
                student_unsup_prob = F.softmax(student_unsup_logits, dim=1)

                # Distance map weighting
                # (if dist_contour_range is 0 then weight will just be 1)
                weight = dist_map_weighting(t_dist_map, batch_u_X, batch_u_X_1)

                # Confidence thresholding
                conf_fac = conf_factor(teacher_unsup_prob)
                mod_fac = conf_fac * weight

                # Compute consistency loss
                if cons_loss_fn == 'bce':
                    aug_loss = robust_binary_crossentropy(student_unsup_prob, teacher_unsup_prob)
                    aug_loss = aug_loss.mean(dim=1) * mod_fac
                    cons_loss = aug_loss.sum() / weight.sum()
                elif cons_loss_fn == 'var':
                    d_aug_loss = student_unsup_prob - teacher_unsup_prob
                    aug_loss = d_aug_loss * d_aug_loss
                    aug_loss = aug_loss.mean(dim=1) * mod_fac
                    cons_loss = aug_loss.sum() / weight.sum()
                elif cons_loss_fn == 'logits_var':
                    d_aug_loss = student_unsup_logits - teacher_unsup_logits
                    aug_loss = d_aug_loss * d_aug_loss
                    aug_loss = aug_loss.mean(dim=1) * mod_fac
                    cons_loss = aug_loss.sum() / weight.sum()
                else:
                    raise ValueError

                # Combine supervised and consistency loss
                loss = sup_loss + cons_loss * cons_weight

                conf_rate = float(conf_fac.sum())
            else:
                loss = sup_loss
                conf_rate = 0.0
                cons_loss = 0.0

            loss.backward()
            student_optimizer.step()
            if teacher_optimizer is not None:
                teacher_optimizer.step()

            batch_sup_loss_accum += float(sup_loss)
            batch_conf_mask_sum_accum += conf_rate
            batch_cons_loss_accum += float(cons_loss)
            batch_N_accum += len(batch_X)

        if batch_N_accum > 0:
            batch_sup_loss_accum /= batch_N_accum
            batch_conf_mask_sum_accum /= batch_N_accum
            batch_cons_loss_accum /= batch_N_accum

        student_net.eval()
        if teacher_net is not None:
            teacher_net.eval()

        # Generate output for plotting
        if save_output and submit_config.run_dir is not None:
            plot_path = os.path.join(submit_config.run_dir, 'epoch_{:05d}.png'.format(epoch + 1))
            cv2.imwrite(plot_path, render_output_image())
        else:
            cv2.imshow('Vis', render_output_image())

            k = cv2.waitKey(1)
            if (k & 255) == 27:
                terminated = True
                break

        t2 = time.time()
        # print('Epoch {}: took {:.3f}s: clf loss={:.6f}'.format(epoch, t2-t1, clf_loss))
        print('Epoch {}: took {:.3f}s: clf loss={:.6f}, conf rate={:.3%}, cons loss={:.6f}'.format(
            epoch+1, t2-t1, batch_sup_loss_accum, batch_conf_mask_sum_accum, batch_cons_loss_accum))

    # Get final score based on all samples
    all_pred_y = []
    with torch.no_grad():
        for (batch_X,) in all_loader:
            batch_X = batch_X.to(torch_device)
            batch_pred_logits = pred_net(batch_X)
            batch_pred_cls = torch.argmax(batch_pred_logits, dim=1)
            all_pred_y.append(batch_pred_cls.detach().cpu().numpy())
    all_pred_y = np.concatenate(all_pred_y, axis=0)
    err_rate = (all_pred_y != ds.y).mean()
    print('FINAL RESULT: Error rate={:.6%} (supervised and unsupervised samples)'.format(err_rate))

    if not save_output:
        # Close output window
        if not terminated:
            cv2.waitKey()

        cv2.destroyAllWindows()


@click.command()
@click.option('--job_desc', type=str, default='')
@click.option('--dataset', type=str, default='spiral')
@click.option('--region_erode_radius', type=int, default=35)
@click.option('--img_noise_std', type=float, default=2.0)
@click.option('--n_sup', type=int, default=10)
@click.option('--balance_classes', is_flag=True, default=False)
@click.option('--seed', type=int, default=12345)
@click.option('--sup_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option('--model', type=click.Choice(['mean_teacher', 'pi', 'pi_onebatch']), default='mean_teacher')
@click.option('--n_hidden', type=int, default=3)
@click.option('--hidden_size', type=int, default=512)
@click.option('--hidden_act', type=click.Choice(['relu', 'lrelu']), default='relu')
@click.option('--norm_layer', type=click.Choice(['none', 'batch_norm', 'weight_norm',
                                                 'spectral_norm', 'group_norm']), default='batch_norm')
@click.option('--perturb_noise_std', type=str, default='6.0')
@click.option('--dist_contour_range', type=float, default=0.0)
@click.option('--conf_thresh', type=float, default=0.97)
@click.option('--conf_avg', is_flag=True, default=False)
@click.option('--cons_weight', type=float, default=10.0)
@click.option('--cons_loss_fn', type=click.Choice(['var', 'bce', 'logits_var']), default='var')
@click.option('--cons_no_dropout', is_flag=True, default=False)
@click.option('--learning_rate', type=float, default=2e-4)
@click.option('--teacher_alpha', type=float, default=0.99)
@click.option('--num_epochs', type=int, default=100)
@click.option('--batch_size', type=int, default=512)
@click.option('--render_cons_grad', is_flag=True, default=False)
@click.option('--render_pred', type=click.Choice(['class', 'prob']), default='prob')
@click.option('--device', type=str, default='cuda:0')
@click.option('--save_output', is_flag=True, default=False)
def experiment(job_desc, dataset, region_erode_radius, img_noise_std, n_sup, balance_classes, seed,
               sup_path, model, n_hidden, hidden_size, hidden_act, norm_layer,
               perturb_noise_std, dist_contour_range,
               conf_thresh, conf_avg,
               cons_weight, cons_loss_fn, cons_no_dropout,
               learning_rate, teacher_alpha,
               num_epochs, batch_size, render_cons_grad, render_pred, device, save_output):
    params = locals().copy()

    train_toy2d.submit(**params)


if __name__ == '__main__':
    experiment()