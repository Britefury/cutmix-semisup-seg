# Usage:
# > sh run_cityscapes_experiments.sh <run_number> <split_seed>
# E.g. to replicate our experiments:
# > sh run_cityscapes_experiments.sh 01 12345
# > sh run_cityscapes_experiments.sh 02 23456
# > sh run_cityscapes_experiments.sh 03 34567
# > sh run_cityscapes_experiments.sh 04 45678
# > sh run_cityscapes_experiments.sh 05 56789

run=${1}
seed=${2}

# Cityscapes training settings
PARAMS_CITYSCAPES_DEEPLAB2I="--dataset=cityscapes --arch=resnet101_deeplab_imagenet --freeze_bn --batch_size=4 --learning_rate=3e-5 --iters_per_epoch=1000 --num_epochs=40"

# Cityscapes augmentation settings
AUG_CITYSCAPES="--crop_size=256,512 --aug_hflip --aug_strong_colour"

# Cityscapes semi-supervised regularizers
REG_SUPERVISED="--cons_weight=0.0"
REG_MASK_CUTOUT="--cons_weight=1.0 --mask_mode=zero --mask_prop_range=0.0:1.0 --conf_thresh=0.97"
REG_MASK_CUTMIX="--cons_weight=1.0 --mask_mode=mix --mask_prop_range=0.5 --conf_thresh=0.97"

# Run the experiments
# Supervised baseline
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=100 ${REG_SUPERVISED} --job_desc=cityscapes_deeplab2i_lr3e-5_sup_100_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=372 ${REG_SUPERVISED} --job_desc=cityscapes_deeplab2i_lr3e-5_sup_372_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=744 ${REG_SUPERVISED} --job_desc=cityscapes_deeplab2i_lr3e-5_sup_744_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=-1 ${REG_SUPERVISED} --job_desc=cityscapes_deeplab2i_lr3e-5_sup_all_run${run} --split_seed=${seed}

# Mask based: CutMix and Cutout
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=100 ${REG_MASK_CUTMIX} --job_desc=cityscapes_deeplab2i_lr3e-5_cutmix_semisup_100_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=372 ${REG_MASK_CUTMIX} --job_desc=cityscapes_deeplab2i_lr3e-5_cutmix_semisup_372_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=744 ${REG_MASK_CUTMIX} --job_desc=cityscapes_deeplab2i_lr3e-5_cutmix_semisup_744_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=-1 ${REG_MASK_CUTMIX} --job_desc=cityscapes_deeplab2i_lr3e-5_cutmix_semisup_all_run${run} --split_seed=${seed}

python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=100 ${REG_MASK_CUTOUT} --job_desc=cityscapes_deeplab2i_lr3e-5_cutout_semisup_100_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=372 ${REG_MASK_CUTOUT} --job_desc=cityscapes_deeplab2i_lr3e-5_cutout_semisup_372_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=744 ${REG_MASK_CUTOUT} --job_desc=cityscapes_deeplab2i_lr3e-5_cutout_semisup_744_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_CITYSCAPES_DEEPLAB2I} ${AUG_CITYSCAPES} --n_sup=-1 ${REG_MASK_CUTOUT} --job_desc=cityscapes_deeplab2i_lr3e-5_cutout_semisup_all_run${run} --split_seed=${seed}

