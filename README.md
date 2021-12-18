# Semi-supervised semantic segmentation using CutMix and Colour Augmentation

Implementations of our papers:

- [Semi-supervised semantic segmentation needs strong, varied perturbations](https://arxiv.org/abs/1906.01916) by
  Geoff French, Samuli Laine, Timo Aila, Michal Mackiewicz and Graham Finlayson
- [Colour augmentation for improved semi-supervised semantic segmentation](https://arxiv.org/abs/2110.04487) by
  Geoff French and Michal Mackiewicz
  

Licensed under MIT license.


## Colour augmentation

Please see our new [paper](https://arxiv.org/abs/2110.04487) for a full discussion, but a summary of our findings can
be found in our [colour augmentation](Colour augmentation.ipynb) Jupyter notebook.


## Requirements

We provide an `environment.yml` file that can be used to re-create a `conda` environment that provides the required
packages:

```
conda env create -f environment.yml
```

Then activate with:

```
conda activate cutmix_semisup_seg
```

(**note**: this will not install the library needed to use the PSPNet architecture; see below)

In general we need:

- Python >= 3.6
- PyTorch >= 1.4
- torchvision 0.5
- OpenCV
- Pillow
- Scikit-image
- Scikit-learn
- click
- tqdm
- Jupyter notebook for the notebooks
- numpy 1.18

#### Requirements for PSPNet

To use the PSPNet architecture (see [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105)
by Zhao et al.), you will need to install the `logits-from_models` branch of
[https://github.com/Britefury/semantic-segmentation-pytorch](https://github.com/Britefury/semantic-segmentation-pytorch):

```
pip install git+https://github.com/Britefury/semantic-segmentation-pytorch.git@logits-from-models
```

## Datasets

You need to:

1. Download/acquire the datsets
2. Write the config file `semantic_segmentation.cfg` giving their paths
3. Convert them if necessary; the CamVid, Cityscapes and ISIC 2017 datasets must be converted
to a ZIP-based format prior to use. You must run the provided conversion utilities to create these ZIP files.

Dataset preparation instructions can be found [here](./DATASETS.md).


## Running the experiments

We provide four programs for running experiments:

- `train_seg_semisup_mask_mt.py`: mask driven consistency loss (the main experiment) 
- `train_seg_semisup_aug_mt.py`: augmentation driven consistency loss; used to attempt to replicate the
 ISIC 2017 baselines of [Li et al.](https://arxiv.org/abs/1808.03887)
- `train_seg_semisup_ict.py`: Interpolation Consistency Training; a baseline for contrast with our main
approach
- `train_seg_semisup_vat_mt.py`: Virtual Adversarial Training adapted for semantic segmentation

They can be configured via command line arguments that are described [here](./CMDLINE_OPTIONS.md).


#### Shell scripts
To replicate our results, we provide shell scripts to run our experiments.

##### Cityscapes
```
> sh run_cityscapes_experiments.sh <run> <split_rng_seed>
```
where `<run>` is the name of the run and `<split_rng_seed>` is an integer RNG seed used to select
the supervised samples. Please see the comments
at the top of `run_cityscapes_experiments.sh` for further explanation.

To re-create the 5 runs we used for our experiments:

```
> sh run_cityscapes_experiments.sh 01 12345
> sh run_cityscapes_experiments.sh 02 23456
> sh run_cityscapes_experiments.sh 03 34567
> sh run_cityscapes_experiments.sh 04 45678
> sh run_cityscapes_experiments.sh 05 56789
```
  
##### Pascal VOC 2012 (augmented)
```
> sh run_pascal_aug_experiments.sh <n_supervised> <n_supervised_txt>
```
where `<n_supervised>` is the number of supervised samples and `<n_supervised_txt>` is that number as text.
Please see the comments at the top of `run_pascal_aug_experiments.sh` for further explanation.

We use the same data split as [Mittal et al.](https://arxiv.org/abs/1908.05724) It is stored in 
`data/splits/pascal_aug/split_0.pkl` that is included in the repo.

##### Pascal VOC 2012 (augmented) with DeepLab v3+
```
> sh run_pascal_aug_deeplab3plus_experiments.sh <n_supervised> <n_supervised_txt>
```

##### ISIC 2017 Segmentation
```
> sh run_isic2017_experiments.sh <run> <split_rng_seed>
```
where `<run>` is the name of the run and `<split_rng_seed>` is an integer RNG seed used to select
the supervised samples. Please see the comments
at the top of `run_isic2017_experiments.sh` for further explanation.

To re-create the 5 runs we used for our experiments: 

```
> sh run_isic2017_experiments.sh 01 12345
> sh run_isic2017_experiments.sh 02 23456
> sh run_isic2017_experiments.sh 07 78901
> sh run_isic2017_experiments.sh 08 89012
> sh run_isic2017_experiments.sh 09 90123
```

In early experiments, we test 10 seeds and selected the middle 5 when ranked in terms of performance,
hence the specific seed choice.
  

## Exploring the input data distribution present in semantic segmentation problems

#### Cluster assumption
First we examine the input data distribution presented by semantic segmentation problems
with a view to determining if the low density separation assumption holds,
in the notebook `Semantic segmentation input data distribution.ipynb`
This notebook also contains the code used to generate the images from Figure 1 in the paper.

#### Inter-class and intra-class variance
Secondly we examine the inter-class and intra-class distance (as a proxy for inter-class and intra-class variance)
in the notebook `Plot inter-class and intra-class distances from files.ipynb`

Note that running the second notebook requires that you generate some data files using the
`intra_inter_class_patch_dist.py` program.


## Toy 2D experiments

The toy 2D experiments used to produce Figure 3 in the paper can be run using the `toy2d_train.py`
program, which is documented [here](./TOY2D.md).

You can re-create the toy 2D experiments by running the `run_toy2d_experiments.sh` shell script:

```
> sh run_toy2d_experiments.sh <run>
```

