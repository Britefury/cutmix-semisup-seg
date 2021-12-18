# Usage:
# > sh run_isic2017_experiments.sh <run_number> <split_seed>
# E.g. to replicate our experiments:
# > sh run_isic2017_experiments.sh 01 12345
# > sh run_isic2017_experiments.sh 02 23456
# > sh run_isic2017_experiments.sh 07 78901
# > sh run_isic2017_experiments.sh 08 89012
# > sh run_isic2017_experiments.sh 09 90123

# Get run number and split RNG seed
run=${1}
seed=${2}

# ISIC 2017 training settings
PARAMS_ISIC2017_DENSEUNET_IMAGENET="--dataset=isic2017 --arch=densenet161unet_imagenet --batch_size=10 --iters_per_epoch=400 --num_epochs=100 --opt_type=sgd --learning_rate=0.1 --sgd_weight_decay=5e-4 --lr_sched=poly --bin_fill_holes"

# ISIC 2017 augmentation settings: 224x224 crop, h/v/hv flip, scale [0.9,1.1], rotate [-45,45]
AUG_ISIC2017_SCLROT="--crop_size=224,224 --aug_hflip --aug_vflip --aug_hvflip --aug_max_scale=1.1 --aug_rot_mag=45.0 --aug_strong_colour"

# ISIC 2017 semi-supervised regularizers
REG_SUPERVISED="--cons_weight=0.0"
REG_AUG_SEMISUP_CW01="--cons_weight=0.1 --conf_thresh=0.97"
REG_ICT01_CW00003="--cons_weight=0.0003 --ict_alpha=0.1 --conf_thresh=0.97"
REG_VAT_ADARAD1_CW0001="--adaptive_vat_radius --vat_radius=1.0 --cons_weight=0.001 --conf_thresh=0.97"
REG_MASK_CUTOUT_CW1="--cons_weight=1.0 --mask_mode=zero --mask_prop_range=0.0:1.0 --conf_thresh=0.97"
REG_MASK_CUTMIX_CW1="--cons_weight=1.0 --mask_mode=mix --mask_prop_range=0.5 --conf_thresh=0.97"

# Run the experiments
# Supervised baselines
python train_seg_semisup_aug_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_SUPERVISED} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_sup_50_run${run} --split_seed=${seed}
python train_seg_semisup_aug_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=-1 ${REG_SUPERVISED} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_sup_all_run${run} --split_seed=${seed}
# Mask based: CutMix and Cutout
python train_seg_semisup_mask_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_MASK_CUTMIX_CW1} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_cutmix_cw1.0_semisup_50_run${run} --split_seed=${seed}
python train_seg_semisup_mask_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_MASK_CUTOUT_CW1} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_cutout_cw1.0_semisup_50_run${run} --split_seed=${seed}
# Augmentation baseline (Li et al.)
python train_seg_semisup_aug_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_AUG_SEMISUP_CW01} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_cw0.1_semisup_50_run${run} --split_seed=${seed}
# ICT
python train_seg_semisup_ict.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_ICT01_CW00003} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_ict0.1_cw0.0003_semisup_50_run${run} --split_seed=${seed}
# VAT
python train_seg_semisup_vat_mt.py ${PARAMS_ISIC2017_DENSEUNET_IMAGENET} ${AUG_ISIC2017_SCLROT} --n_sup=50 ${REG_VAT_ADARAD1_CW0001} --job_desc=isic2017_denseuneti_sgd_lr0.1_wd5e-4_sclrot_vatc_ada1_cw0.001_semisup_50_run${run} --split_seed=${seed}

