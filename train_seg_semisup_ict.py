import job_helper
import click

@job_helper.job('train_seg_semisup_ict', enumerate_job_names=False)
def train_seg_semisup_ict(submit_config: job_helper.SubmitConfig, dataset, model, arch, freeze_bn,
                          opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
                          learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
                          teacher_alpha, bin_fill_holes,
                          crop_size, aug_hflip, aug_vflip, aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,
                          aug_strong_colour, aug_colour_brightness, aug_colour_contrast, aug_colour_saturation, aug_colour_hue,
                          aug_colour_prob, aug_colour_greyscale_prob,
                          ict_alpha, cons_loss_fn, cons_weight, conf_thresh, conf_per_pixel, rampup, unsup_batch_ratio,
                          num_epochs, iters_per_epoch, batch_size,
                          n_sup, n_unsup, n_val, split_seed, split_path, val_seed, save_preds, save_model, num_workers):
    settings = locals().copy()
    del settings['submit_config']

    import os
    import math
    import time
    import itertools
    import numpy as np
    import torch.nn as nn, torch.nn.functional as F
    from architectures import network_architectures
    import torch.utils.data
    import torchvision.transforms as tvt
    from datapipe import datasets
    from datapipe import seg_data, seg_transforms, seg_transforms_cv
    import evaluation
    import optim_weight_ema
    import lr_schedules


    if crop_size == '':
        crop_size = None
    else:
        crop_size = [int(x.strip()) for x in crop_size.split(',')]

    torch_device = torch.device('cuda:0')


    #
    # Load data sets
    #
    ds_dict = datasets.load_dataset(dataset, n_val, val_seed, n_sup, n_unsup, split_seed, split_path)

    ds_src = ds_dict['ds_src']
    ds_tgt = ds_dict['ds_tgt']
    tgt_val_ndx = ds_dict['val_ndx_tgt']
    src_val_ndx = ds_dict['val_ndx_src'] if ds_src is not ds_tgt else None
    test_ndx = ds_dict['test_ndx_tgt']
    sup_ndx = ds_dict['sup_ndx']
    unsup_ndx = ds_dict['unsup_ndx']

    n_classes = ds_src.num_classes
    root_n_classes = math.sqrt(n_classes)

    if bin_fill_holes and n_classes != 2:
        print('Binary hole filling can only be used with binary (2-class) segmentation datasets')
        return

    print('Loaded data')



    # Build network
    NetClass = network_architectures.seg.get(arch)

    student_net = NetClass(ds_src.num_classes).to(torch_device)

    if opt_type == 'adam':
        student_optim = torch.optim.Adam([
            dict(params=student_net.pretrained_parameters(), lr=learning_rate * 0.1),
            dict(params=student_net.new_parameters(), lr=learning_rate)])
    elif opt_type == 'sgd':
        student_optim = torch.optim.SGD([
            dict(params=student_net.pretrained_parameters(), lr=learning_rate * 0.1),
            dict(params=student_net.new_parameters(), lr=learning_rate)],
            momentum=sgd_momentum, nesterov=sgd_nesterov, weight_decay=sgd_weight_decay)
    else:
        raise ValueError('Unknown opt_type {}'.format(opt_type))

    if model == 'mean_teacher':
        teacher_net = NetClass(ds_src.num_classes).to(torch_device)

        for p in teacher_net.parameters():
            p.requires_grad = False

        teacher_optim = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, teacher_alpha)
        eval_net = teacher_net
    elif model == 'pi':
        teacher_net = student_net
        teacher_optim = None
        eval_net = student_net
    else:
        print('Unknown model type {}'.format(model))
        return


    BLOCK_SIZE = student_net.BLOCK_SIZE
    NET_MEAN, NET_STD = seg_transforms.get_mean_std(ds_tgt, student_net)

    if freeze_bn:
        if not hasattr(student_net, 'freeze_batchnorm'):
            raise ValueError('Network {} does not support batchnorm freezing'.format(arch))

    clf_crossent_loss = nn.CrossEntropyLoss(ignore_index=255)

    print('Built network')


    if iters_per_epoch == -1:
        iters_per_epoch = len(unsup_ndx) // batch_size
    total_iters = iters_per_epoch * num_epochs

    lr_epoch_scheduler, lr_iter_scheduler = lr_schedules.make_lr_schedulers(
        optimizer=student_optim, total_iters=total_iters, schedule_type=lr_sched,
        step_epochs=lr_step_epochs, step_gamma=lr_step_gamma, poly_power=lr_poly_power
    )

    # Train data pipeline: transforms
    train_transforms = []

    if crop_size is not None:
        if aug_scale_hung:
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCropScaleHung(crop_size, (0, 0), uniform_scale=not aug_scale_non_uniform))
        elif aug_max_scale != 1.0 or aug_rot_mag != 0.0:
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCropRotateScale(
                crop_size, (0, 0), rot_mag=aug_rot_mag, max_scale=aug_max_scale,
                uniform_scale=not aug_scale_non_uniform, constrain_rot_scale=True))
        else:
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCrop(crop_size, (0, 0)))
    else:
        if aug_scale_hung:
            raise NotImplementedError('aug_scale_hung requires a crop_size')

    if aug_hflip or aug_vflip or aug_hvflip:
        train_transforms.append(
            seg_transforms_cv.SegCVTransformRandomFlip(aug_hflip, aug_vflip, aug_hvflip))

    # Duplicate transforms so far for unsupervised path
    train_unsup_transforms = train_transforms.copy()
    # Flag indicating if the unsupervised batches are expected to be paired
    unsup_paired = False
    if aug_strong_colour:
        colour_xforms = tvt.Compose([
            tvt.RandomApply([
                tvt.ColorJitter(aug_colour_brightness, aug_colour_contrast, aug_colour_saturation, aug_colour_hue)  # not strengthened
            ], p=aug_colour_prob),
            tvt.RandomGrayscale(p=aug_colour_greyscale_prob),
        ])
        train_unsup_transforms.append(seg_transforms.SegTransformToPair())
        train_unsup_transforms.append(seg_transforms_cv.SegCVTransformTVT(colour_xforms))
        unsup_paired = True

    train_transforms.append(seg_transforms_cv.SegCVTransformNormalizeToTensor(NET_MEAN, NET_STD))
    train_unsup_transforms.append(seg_transforms_cv.SegCVTransformNormalizeToTensor(NET_MEAN, NET_STD))

    # Train data pipeline: supervised and unsupervised data sets
    train_sup_ds = ds_src.dataset(labels=True, mask=False, xf=False,
                                  transforms=seg_transforms.SegTransformCompose(train_transforms),
                                  pipeline_type='cv')
    train_unsup_ds = ds_src.dataset(labels=False, mask=True, xf=False,
                                    transforms=seg_transforms.SegTransformCompose(train_unsup_transforms),
                                    pipeline_type='cv')

    collate_fn = seg_data.SegCollate(BLOCK_SIZE)

    # Train data pipeline: data loaders
    sup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(sup_ndx))
    train_sup_loader = torch.utils.data.DataLoader(train_sup_ds, batch_size, sampler=sup_sampler,
                                                   collate_fn=collate_fn, num_workers=num_workers)
    if cons_weight > 0.0:
        unsup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(unsup_ndx))
        train_unsup_loader = torch.utils.data.DataLoader(train_unsup_ds, batch_size, sampler=unsup_sampler,
                                                         collate_fn=collate_fn, num_workers=num_workers)
    else:
        train_unsup_loader = None


    # Eval pipeline
    src_val_loader, tgt_val_loader, test_loader = datasets.eval_data_pipeline(
        ds_src, ds_tgt, src_val_ndx, tgt_val_ndx, test_ndx, batch_size, collate_fn, NET_MEAN, NET_STD, num_workers)


    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))

    # Report dataset size
    print('Dataset:')
    print('len(sup_ndx)={}'.format(len(sup_ndx)))
    print('len(unsup_ndx)={}'.format(len(unsup_ndx)))
    if ds_src is not ds_tgt:
        print('len(src_val_ndx)={}'.format(len(tgt_val_ndx)))
        print('len(tgt_val_ndx)={}'.format(len(tgt_val_ndx)))
    else:
        print('len(val_ndx)={}'.format(len(tgt_val_ndx)))
    if test_ndx is not None:
        print('len(test_ndx)={}'.format(len(test_ndx)))

    if n_sup != -1:
        print('sup_ndx={}'.format(sup_ndx.tolist()))


    # Track mIoU for early stopping
    best_tgt_miou = None
    best_epoch = 0

    eval_net_state = {key: value.detach().cpu().numpy() for key, value in eval_net.state_dict().items()}

    # Create iterators
    train_sup_iter = iter(train_sup_loader)
    train_unsup_iter = iter(train_unsup_loader) if train_unsup_loader is not None else None


    iter_i = 0
    print('Training...')
    for epoch_i in range(num_epochs):
        if lr_epoch_scheduler is not None:
            lr_epoch_scheduler.step(epoch_i)

        t1 = time.time()

        if rampup > 0:
            ramp_val = network_architectures.sigmoid_rampup(epoch_i, rampup)
        else:
            ramp_val = 1.0

        student_net.train()
        if teacher_net is not student_net:
            teacher_net.train()

        if freeze_bn:
            student_net.freeze_batchnorm()
            if teacher_net is not student_net:
                teacher_net.freeze_batchnorm()

        sup_loss_acc = 0.0
        consistency_loss_acc = 0.0
        conf_rate_acc = 0.0
        n_sup_batches = 0
        n_unsup_batches = 0


        src_val_iter = iter(src_val_loader) if src_val_loader is not None else None
        tgt_val_iter = iter(tgt_val_loader) if tgt_val_loader is not None else None

        for sup_batch in itertools.islice(train_sup_iter, iters_per_epoch):
            if lr_iter_scheduler is not None:
                lr_iter_scheduler.step(iter_i)
            student_optim.zero_grad()

            #
            # Supervised branch
            #

            batch_x = sup_batch['image'].to(torch_device)
            batch_y = sup_batch['labels'].to(torch_device)

            logits_sup = student_net(batch_x)
            sup_loss = clf_crossent_loss(logits_sup, batch_y[:,0,:,:])
            sup_loss.backward()

            if cons_weight > 0.0:
                for _ in range(unsup_batch_ratio):
                    #
                    # Unsupervised branch
                    #

                    # Mix mode: batch consists of paired unsupervised samples
                    unsup_batch0 = next(train_unsup_iter)
                    unsup_batch1 = next(train_unsup_iter)
                    if unsup_paired:
                        # The teacher path should come from sample 0 that has weaker
                        # augmentation (no colour augmentation), where the student should
                        # use sample 1 that has stronger augmentation

                        batch_ux0_tea = unsup_batch0['sample0']['image'].to(torch_device)
                        batch_ux0_stu = unsup_batch0['sample1']['image'].to(torch_device)
                        batch_um0 = unsup_batch0['sample0']['mask'].to(torch_device)
                        batch_ux1_tea = unsup_batch1['sample0']['image'].to(torch_device)
                        batch_ux1_stu = unsup_batch1['sample1']['image'].to(torch_device)
                        batch_um1 = unsup_batch1['sample0']['mask'].to(torch_device)
                    else:
                        batch_ux0_tea = unsup_batch0['image'].to(torch_device)
                        batch_ux0_stu = batch_ux0_tea
                        batch_um0 = unsup_batch0['mask'].to(torch_device)
                        batch_ux1_tea = unsup_batch1['image'].to(torch_device)
                        batch_ux1_stu = batch_ux1_tea
                        batch_um1 = unsup_batch1['mask'].to(torch_device)

                    # batch_um0 and batch_um1 are masks that are 1 for valid pixels, 0 for invalid pixels.
                    # They are used later on to scale the consistency loss, so that consistency loss is
                    # only computed for valid pixels.
                    # Explanation:
                    # When using geometric augmentations such as rotations, some pixels in the training
                    # crop may come from outside the bounds of the input image. These pixels will have a value
                    # of 0 in these masks. Similarly, when using scaled crops, the size of the crop
                    # from the input image that must be scaled to the size of the training crop may be
                    # larger than one/both of the input image dimensions. Pixels in the training crop
                    # that arise from outside the input image bounds will once again be given a value
                    # of 0 in these masks.

                    # ICT mix factors
                    ict_mix_factors = np.random.beta(ict_alpha, ict_alpha, size=(len(batch_ux0_tea), 1, 1, 1))
                    ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float, device=torch_device)

                    # Mix images
                    batch_ux_stu_mixed = batch_ux0_stu * (1.0 - ict_mix_factors) + batch_ux1_stu * ict_mix_factors
                    batch_um_mixed = batch_um0 * (1.0 - ict_mix_factors) + batch_um1 * ict_mix_factors

                    # Get teacher predictions for original images
                    with torch.no_grad():
                        logits_u0_tea = teacher_net(batch_ux0_tea).detach()
                        logits_u1_tea = teacher_net(batch_ux1_tea).detach()
                    # Get student prediction for mixed image
                    logits_cons_stu = student_net(batch_ux_stu_mixed)

                    # Logits -> probs
                    prob_u0_tea = F.softmax(logits_u0_tea, dim=1)
                    prob_u1_tea = F.softmax(logits_u1_tea, dim=1)
                    prob_cons_stu = F.softmax(logits_cons_stu, dim=1)

                    # Mix teacher predictions using same mask
                    # It makes no difference whether we do this with logits or probabilities as
                    # the mask pixels are either 1 or 0
                    logits_cons_tea = logits_u0_tea * (1 - ict_mix_factors) + logits_u1_tea * ict_mix_factors
                    prob_cons_tea = prob_u0_tea * (1 - ict_mix_factors) + prob_u1_tea * ict_mix_factors

                    loss_mask = batch_um_mixed

                    # Confidence thresholding
                    if conf_thresh > 0.0:
                        # Compute probabilities then confidence of each teacher prediction
                        prob_u0_tea = F.softmax(logits_u0_tea, dim=1)
                        prob_u1_tea = F.softmax(logits_u1_tea, dim=1)
                        conf_u0_tea = prob_u0_tea.max(dim=1, keepdim=True)[0]
                        conf_u1_tea = prob_u1_tea.max(dim=1, keepdim=True)[0]
                        # Mix confidences
                        conf_tea = conf_u0_tea * (1 - ict_mix_factors) + conf_u1_tea * ict_mix_factors
                        # Compute confidence mask
                        conf_mask = (conf_tea >= conf_thresh).float()[:, None, :, :]
                        # Record rate for reporting
                        conf_rate_acc += float(conf_mask.mean())
                        # Average confidence mask if requested
                        if not conf_per_pixel:
                            conf_mask = conf_mask.mean()

                        loss_mask = loss_mask * conf_mask
                    elif rampup > 0:
                        conf_rate_acc += ramp_val

                    # Compute per-pixel consistency loss
                    # Note that the way we aggregate the loss across the class/channel dimension (1)
                    # depends on the loss function used. Generally, summing over the class dimension
                    # keeps the magnitude of the gradient of the loss w.r.t. the logits
                    # nearly constant w.r.t. the number of classes. When using logit-variance,
                    # dividing by `sqrt(num_classes)` helps.
                    if cons_loss_fn == 'var':
                        delta_prob = prob_cons_stu - prob_cons_tea
                        consistency_loss = delta_prob * delta_prob
                        consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                    elif cons_loss_fn == 'logits_var':
                        delta_logits = logits_cons_stu - logits_cons_tea
                        consistency_loss = delta_logits * delta_logits
                        consistency_loss = consistency_loss.sum(dim=1, keepdim=True) / root_n_classes
                    elif cons_loss_fn == 'logits_smoothl1':
                        consistency_loss = F.smooth_l1_loss(logits_cons_stu,
                                                            logits_cons_tea, reduce=False)
                        consistency_loss = consistency_loss.sum(dim=1, keepdim=True) / root_n_classes
                    elif cons_loss_fn == 'bce':
                        consistency_loss = network_architectures.robust_binary_crossentropy(prob_cons_stu,
                                                                                            prob_cons_tea)
                        consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                    elif cons_loss_fn == 'kld':
                        consistency_loss = F.kl_div(F.log_softmax(logits_cons_stu, dim=1), prob_cons_tea, reduce=False)
                        consistency_loss = consistency_loss.sum(dim=1, keepdim=True)
                    else:
                        raise ValueError('Unknown consistency loss function {}'.format(cons_loss_fn))

                    # Apply consistency loss mask and take the mean over pixels and images
                    consistency_loss = (consistency_loss * loss_mask).mean()

                    # Modulate with rampup if desired
                    if rampup > 0:
                        consistency_loss = consistency_loss * ramp_val

                    # Weight the consistency loss and back-prop
                    unsup_loss = consistency_loss * cons_weight
                    unsup_loss.backward()

                    consistency_loss_acc += float(consistency_loss.detach())

                    n_unsup_batches += 1

            student_optim.step()
            if teacher_optim is not None:
                teacher_optim.step()

            sup_loss_acc += float(sup_loss.detach())
            n_sup_batches += 1
            iter_i += 1


        sup_loss_acc /= n_sup_batches
        if n_unsup_batches > 0:
            consistency_loss_acc /= n_unsup_batches
            conf_rate_acc /= n_unsup_batches

        eval_net.eval()

        if ds_src is not ds_tgt:
            src_iou_eval = evaluation.EvaluatorIoU(ds_src.num_classes, bin_fill_holes)
            with torch.no_grad():
                for batch in src_val_iter:
                    batch_x = batch['image'].to(torch_device)
                    batch_y = batch['labels'].numpy()

                    logits = eval_net(batch_x)
                    pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                    for sample_i in range(len(batch_y)):
                        src_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

            src_iou = src_iou_eval.score()
            src_miou = src_iou.mean()
        else:
            src_iou_eval = src_iou = src_miou = None

        tgt_iou_eval = evaluation.EvaluatorIoU(ds_tgt.num_classes, bin_fill_holes)
        with torch.no_grad():
            for batch in tgt_val_iter:
                batch_x = batch['image'].to(torch_device)
                batch_y = batch['labels'].numpy()

                logits = eval_net(batch_x)
                pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                for sample_i in range(len(batch_y)):
                    tgt_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

        tgt_iou = tgt_iou_eval.score()
        tgt_miou = tgt_iou.mean()

        t2 = time.time()

        if ds_src is not ds_tgt:
            print('Epoch {}: took {:.3f}s, TRAIN clf loss={:.6f}, consistency loss={:.6f}, conf rate={:.3%}, '
                  'SRC VAL mIoU={:.3%}, TGT VAL mIoU={:.3%}'.format(
                epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, conf_rate_acc, src_miou, tgt_miou))
            print('-- SRC {}'.format(', '.join(['{:.3%}'.format(x) for x in src_iou])))
            print('-- TGT {}'.format(', '.join(['{:.3%}'.format(x) for x in tgt_iou])))
        else:
            print('Epoch {}: took {:.3f}s, TRAIN clf loss={:.6f}, consistency loss={:.6f}, conf rate={:.3%}, VAL mIoU={:.3%}'.format(
                epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, conf_rate_acc, tgt_miou))
            print('-- {}'.format(', '.join(['{:.3%}'.format(x) for x in tgt_iou])))


    if save_model:
        model_path = os.path.join(submit_config.run_dir, "model.pth")
        torch.save(eval_net, model_path)

    if save_preds:
        out_dir = os.path.join(submit_config.run_dir, 'preds')
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            for batch in tgt_val_loader:
                batch_x = batch['image'].to(torch_device)
                batch_ndx = batch['index'].numpy()

                logits = eval_net(batch_x)
                pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                for sample_i, sample_ndx in enumerate(batch_ndx):
                    ds_tgt.save_prediction_by_index(out_dir, pred_y[sample_i].astype(np.uint32), sample_ndx)
    else:
        out_dir = None

    if test_loader is not None:
        test_iou_eval = evaluation.EvaluatorIoU(ds_tgt.num_classes, bin_fill_holes)
        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch['image'].to(torch_device)
                if 'labels' in batch:
                    batch_y = batch['labels'].numpy()
                else:
                    batch_y = None
                batch_ndx = batch['index'].numpy()

                logits = eval_net(batch_x)
                pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                for sample_i, sample_ndx in enumerate(batch_ndx):
                    if save_preds:
                        ds_tgt.save_prediction_by_index(out_dir, pred_y[sample_i].astype(np.uint32), sample_ndx)
                    if batch_y is not None:
                        test_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

        test_iou = test_iou_eval.score()
        test_miou = test_iou.mean()

        print('FINAL TEST: mIoU={:.3%}'.format(test_miou))
        print('-- TEST {}'.format(', '.join(['{:.3%}'.format(x) for x in test_iou])))



@click.command()
@click.option('--job_desc', type=str, default='')
@click.option('--dataset', type=click.Choice(['camvid', 'cityscapes', 'pascal', 'pascal_aug', 'isic2017']),
              default='pascal_aug')
@click.option('--model', type=click.Choice(['mean_teacher', 'pi']), default='mean_teacher')
@click.option('--arch', type=str, default='resnet101_deeplab_imagenet')
@click.option('--freeze_bn', is_flag=True, default=False)
@click.option('--opt_type', type=click.Choice(['adam', 'sgd']), default='adam')
@click.option('--sgd_momentum', type=float, default=0.9)
@click.option('--sgd_nesterov', is_flag=True, default=True)
@click.option('--sgd_weight_decay', type=float, default=5e-4)
@click.option('--learning_rate', type=float, default=1e-4)
@click.option('--lr_sched', type=click.Choice(['none', 'stepped', 'cosine', 'poly']), default='none')
@click.option('--lr_step_epochs', type=str, default='')
@click.option('--lr_step_gamma', type=float, default=0.1)
@click.option('--lr_poly_power', type=float, default=0.9)
@click.option('--teacher_alpha', type=float, default=0.99)
@click.option('--bin_fill_holes', is_flag=True, default=False)
@click.option('--crop_size', type=str, default='321,321')
@click.option('--aug_hflip', is_flag=True, default=False)
@click.option('--aug_vflip', is_flag=True, default=False)
@click.option('--aug_hvflip', is_flag=True, default=False)
@click.option('--aug_scale_hung', is_flag=True, default=False)
@click.option('--aug_max_scale', type=float, default=1.0)
@click.option('--aug_scale_non_uniform', is_flag=True, default=False)
@click.option('--aug_rot_mag', type=float, default=0.0)
@click.option('--aug_strong_colour', is_flag=True, default=False)
@click.option('--aug_colour_brightness', type=float, default=0.4)
@click.option('--aug_colour_contrast', type=float, default=0.4)
@click.option('--aug_colour_saturation', type=float, default=0.4)
@click.option('--aug_colour_hue', type=float, default=0.1)
@click.option('--aug_colour_prob', type=float, default=0.8)
@click.option('--aug_colour_greyscale_prob', type=float, default=0.2)
@click.option('--ict_alpha', type=float, default=0.1)
@click.option('--cons_loss_fn', type=click.Choice(['var', 'bce', 'kld', 'logits_var', 'logits_smoothl1']), default='var')
@click.option('--cons_weight', type=float, default=0.3)
@click.option('--conf_thresh', type=float, default=0.97)
@click.option('--conf_per_pixel', is_flag=True, default=False)
@click.option('--rampup', type=int, default=-1)
@click.option('--unsup_batch_ratio', type=int, default=1)
@click.option('--num_epochs', type=int, default=300)
@click.option('--iters_per_epoch', type=int, default=-1)
@click.option('--batch_size', type=int, default=10)
@click.option('--n_sup', type=int, default=100)
@click.option('--n_unsup', type=int, default=-1)
@click.option('--n_val', type=int, default=-1)
@click.option('--split_seed', type=int, default=12345)
@click.option('--split_path', type=click.Path(readable=True, exists=True))
@click.option('--val_seed', type=int, default=131)
@click.option('--save_preds', is_flag=True, default=False)
@click.option('--save_model', is_flag=True, default=False)
@click.option('--num_workers', type=int, default=4)
def experiment(job_desc, dataset, model, arch, freeze_bn,
               opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
               learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
               teacher_alpha, bin_fill_holes,
               crop_size, aug_hflip, aug_vflip, aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,
               aug_strong_colour, aug_colour_brightness, aug_colour_contrast, aug_colour_saturation, aug_colour_hue,
               aug_colour_prob, aug_colour_greyscale_prob,
               ict_alpha, cons_loss_fn, cons_weight, conf_thresh, conf_per_pixel, rampup, unsup_batch_ratio,
               num_epochs, iters_per_epoch, batch_size,
               n_sup, n_unsup, n_val, split_seed, split_path, val_seed, save_preds, save_model, num_workers):
    params = locals().copy()

    train_seg_semisup_ict.submit(**params)

if __name__ == '__main__':
    experiment()