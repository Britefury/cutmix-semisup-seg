# Experiment command-line arguments


### General command line options

These options apply to all experiments. For options for specific experiments (mask based consistency,
augmentation based consistency, ICT, VAT), see below.

###### Output and dataset options
- `--job_desc`: provide a job description/name. For example, running the `train_seg_semisup_mask_mt.py`
with `--job_desc=test_a_1` program will save its log file to `results/train_seg_semisup_mask_mt/log_test_a_1.txt`
and models and predictions will be saved to the directory `results/train_seg_semisup_mask_mt/test_a_1`.
- `--dataset` *[default=pascal_aug]*: select the dataset to train on:, one
    of `camvid`,`cityscapes`,`pascal`,`pascal_aug`  (Pascal VOC 2012 augmented with SBD),`isic2017`
- `--bin_fill_holes`: flag to enable hole filling for foreground class. Only usable for binary segmentation
    tasks e.g. ISIC 2017 segmentation. Used for ISIC 2017 experiments.
- `--save_preds`: if enabled, after training the predictions for validation and test samples will be
    saved in the output directory (see `--job_desc`)
- `--save_model`: if enabled, after training the model will be saved in the output directory
    (see `--job_desc`)

###### Model and architecture options
- `--model` *[default=mean_teacher]*: select the consistency model:
    - `mean_teacher` use the Mean Tecaher model of [Tarvainen et al.](https://arxiv.org/abs/1703.01780)
    - `pi` use the Pi-model of [Laine et al.](https://arxiv.org/abs/1610.02242)
- `--arch` *[default=resnet101_deeplab_imagenet]*: select the model architecture:
    - `resnet50unet_imagenet`: ResNet-50 based U-net with ImageNet classification pre-training
    - `resnet101unet_imagenet`: ResNet-101 based U-net with ImageNet classification pre-training
    - `densenet161unet`: DenseNet-161 based U-Net, randomly initialised
    - `densenet161unet_imagenet`: DenseNet-161 based U-Net, with ImageNet classification pre-training
    - `resnet101_deeplab_coco`: ResNet-101 based DeepLab v2, with CoCo semantic segmentation pre-training
    - `resnet101_deeplab_imagenet` *[default]*: ResNet-101 based DeepLab v2, with ImageNet classification pre-training
    - `resnet101_deeplab_imagenet_mittal_std`: ResNet-101 based DeepLab v2, with ImageNet classification pre-training,
        using the mean and std-dev used for normalization by Mittal et al.
    - `resnet101_deeplabv3_coco`:
        [torchvision ResNet-101 based DeepLab v3](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation)
        with CoCo semantic segmentation pre-training
    - `resnet101_deeplabv3_imagenet`:
        [torchvision ResNet-101 based DeepLab v3](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation)
        with ImageNet classification pre-training
    - `resnet101_deeplabv3plus_imagenet`: ResNet-101 based DeepLab v3+, with ImageNet classification pre-training
    - `resnet101_pspnet_imagenet`: ResNet-101 based PSP-net
        (see [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) by Zhao et al.),
        with ImageNet classification pre-training. To use this architecture you need to install our modified
        version of MIT CSAIL's [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
        library. Grab the `logits-from_models` branch of
        [https://github.com/Britefury/semantic-segmentation-pytorch](https://github.com/Britefury/semantic-segmentation-pytorch)
- `--freeze_bn`: flag to enable freezing of batch-norm layers. Use for DeepLab models, or for `resnet50unet_imagenet`
    if using a batch size of 1

###### Learning rate and optimizer options
- `--opt_type` *[default=adam]*: optimizer type; one of `sgd`, `adam` *[default]*
- `--sgd_momentum` *[default=0.9]*: set momentum if using SGD optimizer 
- `--sgd_nesterov`: flag to enable Nesterov momentum if using SGD optimizer 
- `--sgd_weight_decay` *[default=5e-4]*: set weight decay if using SGD optimizer 
- `--learning_rate` *[default=1e-4]*: set learning rate (use `3e-5` for DeepLab v2 models, `1e-5` for DeepLab v3+,
    `0.1` with SGD optimizer for `densenet161unet_imagenet` for `isic2017` dataset)
- `--lr_sched` *[default=none]*: learning rate scheduler type
    - `none`: no LR schedule
    - `stepped`: stepped LR schedule (control with `--lr_step_epochs` and `--lr_step_gamma` options)
    - `cosine` cosine schedule
    - `poly` polynomial schedule, control exponent with `--lr_poly_power`
- `--lr_step_epochs`: stepped LR schedule step epochs as a Python list, e.g. `--lr_step_epochs=[30,60,80]` will
    change the learning rate at epochs 30, 60 and 80
- `--lr_step_gamma` *[default=0.1]*: stepped LR schedule gamma; reduce the larning rate by this factor at each step
- `--lr_poly_power` *[default=0.9]*: polynomial LR schedule gamma; scale learning rate by `p^(1-(iter/max_iters))`

###### Augmentation options
- `--crop_size`: size of crop to extract during training, as `H,W` e.g. `--crop_size=321,321`. Should be provided.
- `--aug_hflip`: augmentation: enable horizontal flip
- `--aug_vflip`: augmentation: enable vertical flip
- `--aug_hvflip`: augmentation: enable diagonal flip (swap X and Y axes)
- `--aug_scale_hung`: augmentation: enable scaling used by [Hung et al.](https://arxiv.org/abs/1802.07934)
    (scale factor chosen randomly between `0.5` and `1.5` in increments of `0.1`).
- `--aug_max_scale` *[default=1.0]*: augmentation: enable random scale augmentation; scale factor chosen in
    range `[1/aug_max_scale, aug_max_scale]` from log-uniform distribution
    (overriden by `--aug_scale_hung` is used)
- `--aug_scale_non_uniform`: augmentation: enable non-uniform scaling (compatible with both `--aug_scale_hung`
    and `--aug_max_scale`)
- `--aug_rot_mag` *[default=0.0]*: augmentation: random rotation magnitude in degrees; rotate by angle chosen
    from range `[-aug_rot_mag, aug_rot_mag]` (disabled by `--aug_scale_hung`)
- `--aug_strong_colour`: augmentation: enable colour augmentation on strong/student side of consistency loss.
    Colour augmentation consists of applying colour jitter -- with probability determined by the
    `--aug_colour_prob` option -- that consists of jittering the brightness, contrast, saturation and hue
    by the `--aug_colour_brightness`, `--aug_colour_contrast`, `--aug_colour_saturation` and
    `--aug_colour_hue` settings respectively, followed by converting to greyscale with probability
    `--aug_colour_greyscale_prob`. Replicates the procedure used in the
    [MoCo model of He et al.](https://arxiv.org/abs/1911.05722) 
- `--aug_colour_brightness` *[default=0.4]*: colour augmentation: brightness jitter strength
- `--aug_colour_contrast` *[default=0.4]*: colour augmentation: contrast jitter strength
- `--aug_colour_saturation` *[default=0.4]*: colour augmentation: saturation jitter strength
- `--aug_colour_hue` *[default=0.1]*: colour augmentation: hue jitter strength
- `--aug_colour_prob` *[default=0.8]*: colour augmentation: probability of colour jitter
- `--aug_colour_greyscale_prob` *[default=0.2]*: colour augmentation: probability of greyscale

###### Consistency loss and Mean Teacher options
- `--teacher_alpha` *[default=0.99]*: EMA alpha used to update teacher network when using the mean teacher model 
- `--cons_loss_fn`: consistency loss function:
    - `var` *[default for all experiments except VAT]* squared error between predicted probabilities
    - `bce`: binary cross entropy, using teacher predictions as target
    - `kld` *[default for VAT experiment]*: KL-divergence
    - `logits_var`: squared error between predicted pre-softmax logits
    - `logits_smoothl1` *[not available for VAT experiment]*:sSmooth L1-loss between predicted pre-softmax logits
- `--cons_weight` *[default=1.0, 0.3 for ICT experiment]*: consistency loss weight
- `--conf_thresh` *[default=0.97]*: confidence threshold
- `--conf_per_pixel`: flag to enable applying confidence threshold per pixel, otherwise averages the confidence
    mask
- `--rampup` *[default=0]*: Ramp up the consistency loss weight over the specified number of epochs using
    the sigmoid function specified in [Laine et al](https://arxiv.org/abs/1610.02242).
    Only works with `--conf_thresh=0`.
- `--unsup_batch_ratio` *[default=1]*: for each supervised batch, process this number of unsupervised batches. This
    proved to be successful for semi-supervised classification in
    [UDA by Xie et al.](https://arxiv.org/abs/1904.12848) and
    [FixMatch by Sohn et al.](https://arxiv.org/abs/2001.07685). Tests with CamVid yielded limited success for
    segmentation. We didn't try this with other datasets though.

###### Batch size and training options
- `--num_epochs` *[default=300]*: number of epochs to train for
- `--iters_per_epoch` *[default=-1]*: number of iterations per epoch. If `-1` is given, it will be the number of
    mini-batches required to cover the training set
- `--batch_size` *[default=10]*: the mini-batch size 
- `--num_workers` *[default=4]*: the number of worker processes used to load data batches in the background

###### Dataset split options
- `--n_sup` *[default=100]*: the number of supervised samples to use during training. These will be randomly
    selected from the training set, using the random seed provided using `--split_seed` to initialise the
    RNG. Alternative, if `--split_path` is provided, the first `n_sup` samples will be selected
    from the array of indices loaded from the specified file
- `--n_unsup` *[default=-1]*: the number of unsupervised samples to use during training. If `-1` is given
    use all training samples
- `--n_val` *[default=-1]*: the number of samples used for validation. If the dataset provides separate
    validation and test sets (e.g. CamVid) this will be ignored. If `-1` is the provided
    validation/test set will be used as the validation set. If a value is provided, `n_val` samples
    will be randomly selected, using `--val_seed` to initialise the RNG. If `--split_path` is provided,
    the last `n_val` samples in the index array will be used.
- `--split_seed` *[default=12345]*: the seed used to initialise the RNG used to select supervised samples
- `--val_seed` *[default=131]*: the seed used to initialise the RNG used to select validation samples
- `--split_path`: give the path of a pickle (`.pkl`) file from which an index array will be loaded.
    This index array will be used to select supervised and validation samples, rather than an RNG.
    Validation samples will be taken from the end, supervised samples from the start.
    
    

### Options for mask based consistency

These options apply to the `train_seg_semisup_mask_mt.py` program


- `--mask_mode` *[default=mix]*: masking mode
    - `zero`: multiply input images by mask, clearing masked regions to zero
    - `mix`: use mask to blend pairs of input images
- `--mask_prop_range` *[default=0.5]*: mask proportion range; the proportion of the mask with a 1 value
    will be drawn from this range. Either a single value for a fixed proportion (e.g. `0.5`) or a range
    separated by a colon (e.g. `0.0:1.0`).
- `--boxmask_n_boxes` *[default=1]*: number of boxes to draw into the mask. Note that boxes are
    XOR'ed with one another
- `--boxmask_fixed_aspect_ratio`: forces all boxes to have an aspect ration that is the same as the image crop
    (see `--crop_size`). Enable this to precisely replicate [Cutout](https://arxiv.org/abs/1708.04552) or
    [CutMix](https://arxiv.org/abs/1905.04899).
- `--boxmask_by_size`: if enabled, the mask proportion will determine the box edge length, rather than
    the area it covers
- `--boxmask_outside_bounds`: if enabled, box centres will be selected such that part of the box may
    lie outside the bounds of the image crop. Enable this to precisely replicate
    [Cutout](https://arxiv.org/abs/1708.04552) or [CutMix](https://arxiv.org/abs/1905.04899).
- `--boxmask_no_invert`: if enabled, boxes will have a value of 1 against a background of 0, rather
    than the other way around.
    
                


### Options for augmentation driven consistency

These options apply to the `train_seg_semisup_aug_mt.py` program


- `--aug_offset_range` *[default=16]*: augmentation: the centres of the two crops extracted from an unsupervised
    image will be offset from eachother in the range `[-aug_offset_range, aug_offset_range]` 
- `--aug_free_scale_rot`: augmentation: the two crops can have different scale and rotation; only applies to
    `--aug_max_scale` and `--aug_rot_mag`; does not affect `--aug_scale_hung`
     


### Options for Interpolation Consistency Training (ICT)

These options apply to the `train_seg_semisup_ict.py` program


- `--ict_alpha` *[default=0.1]*: alpha value used to determine shape of Beta distribution used to draw blending
    factors
          


### Options for Virtual Adversarial Training (VAT)

These options apply to the `train_seg_semisup_vat_mt.py` program


- `--vat_radius` *[default=0.5]*: the radius that the adversarial perturbation is scaled by. By default
    this radius is `vat_radius * sqrt(H*W*C)` where `H` and `W` are the crop size and `C` is the number
    of input channels.
- `--adaptive_vat_radius`: if enabled, scale the VAT radius adaptively according to the content of each
    unsupervised image; radius is `vat_radius * (|dI/dx| + |dI/dy|) * 0.5`, where `dI/dx` and `dI/dy` are the
    horizontal and vertical gradient images respectively
- `--vat_dir_from_student`: if enabled, use the student network to estimate the perturbation direction
    rather than the teacher (only when using mean teacher model)
     


        
