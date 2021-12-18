# Usage:
# > sh run_pascal_aug_experiments.sh <num_supervised_samples> <num_supervised_as_text_for_job_description>
# E.g. to replicate our experiments:
# > sh run_pascal_aug_experiments.sh 106 106
# > sh run_pascal_aug_experiments.sh 212 212
# > sh run_pascal_aug_experiments.sh 529 529
# > sh run_pascal_aug_experiments.sh 1323 1323
# > sh run_pascal_aug_experiments.sh -1 all

n_sup=${1}
n_sup_txt=${2}

# Pascal VOC 2012 training settings
PARAMS_PASCALAUG_DEEPLAB3PLUSI="--dataset=pascal_aug --arch=resnet101_deeplabv3plus_imagenet --freeze_bn --batch_size=10 --learning_rate=1e-5 --iters_per_epoch=1000 --num_epochs=40 --split_path=./data/splits/pascal_aug/split_0.pkl"

# Pascal VOC 2012 augmentation settings
AUG_PASCAL="--crop_size=321,321 --aug_hflip --aug_scale_hung --aug_strong_colour"

# Pascal VOC 2012 semi-supervised regularizers
REG_SUPERVISED="--cons_weight=0.0"
REG_MASK_CUTMIX="--cons_weight=1.0 --mask_mode=mix --mask_prop_range=0.5 --conf_thresh=0.97"

# Run the experiments
# Supervised baseline
python train_seg_semisup_mask_mt.py ${PARAMS_PASCALAUG_DEEPLAB3PLUSI} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_SUPERVISED} --job_desc=pascalaug_deeplab3plusi_lr1e-5_sup_${n_sup_txt}_split0
# Mask based: CutMix
python train_seg_semisup_mask_mt.py ${PARAMS_PASCALAUG_DEEPLAB3PLUSI} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_MASK_CUTMIX} --job_desc=pascalaug_deeplab3plusi_lr1e-5_cutmix_semisup_${n_sup_txt}_split0

