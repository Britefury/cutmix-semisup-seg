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
PARAMS_PASCALAUG_DEEPLAB2I="--dataset=pascal_aug --arch=resnet101_deeplab_imagenet --freeze_bn --batch_size=10 --learning_rate=3e-5 --iters_per_epoch=1000 --num_epochs=40 --split_path=./data/splits/pascal_aug/split_0.pkl"

# Pascal VOC 2012 augmentation settings
AUG_PASCAL="--crop_size=321,321 --aug_hflip --aug_scale_hung --aug_strong_colour"

# Pascal VOC 2012 semi-supervised regularizers
REG_SUPERVISED="--cons_weight=0.0"
REG_MASK_CUTOUT="--cons_weight=1.0 --mask_mode=zero --mask_prop_range=0.0:1.0 --conf_thresh=0.97"
REG_MASK_CUTMIX="--cons_weight=1.0 --mask_mode=mix --mask_prop_range=0.5 --conf_thresh=0.97"
REG_ICT01="--cons_weight=1.0 --ict_alpha=0.1 --conf_thresh=0.97"
REG_AUG_SEMISUP="--cons_weight=1.0 --conf_thresh=0.97"
REG_VAT_ADARAD1_CW01="--adaptive_vat_radius --vat_radius=1.0 --cons_weight=0.1 --conf_thresh=0.97"

# Run the experiments
# Supervised baseline
python train_seg_semisup_mask_mt.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_SUPERVISED} --job_desc=pascalaug_deeplab2i_lr3e-5_sup_${n_sup_txt}_split0
# Mask based: CutMix and Cutout
python train_seg_semisup_mask_mt.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_MASK_CUTMIX} --job_desc=pascalaug_deeplab2i_lr3e-5_cutmix_semisup_${n_sup_txt}_split0
python train_seg_semisup_mask_mt.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_MASK_CUTOUT} --job_desc=pascalaug_deeplab2i_lr3e-5_cutout_semisup_${n_sup_txt}_split0
# Augmentation
python train_seg_semisup_aug_mt.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_AUG_SEMISUP} --job_desc=pascalaug_deeplab2i_lr3e-5_aug_cw0.003_semisup_${n_sup_txt}_split0
# ICT
python train_seg_semisup_ict.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_ICT01} --job_desc=pascalaug_deeplab2i_lr3e-5_ict0.1_cw0.01_semisup_${n_sup_txt}_split0
# VAT
python train_seg_semisup_vat_mt.py ${PARAMS_PASCALAUG_DEEPLAB2I} ${AUG_PASCAL} --n_sup=${n_sup} ${REG_VAT_ADARAD1_CW01} --job_desc=pascalaug_deeplab2i_lr3e-5_vatc_ada1_cw0.1_semisup_${n_sup_txt}_split0

