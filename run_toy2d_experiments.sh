run=${1}

python toy2d_train.py --job_desc=continuous_semisup_run${run} --dataset=img:data/toy2d/curve_mask_v3.png --sup_path=data/toy2d/curve_mask_v3_35.pkl --region_erode_radius=0 --norm_layer=none --cons_no_dropout --cons_loss_fn=logits_var --cons_weight=1.0 --perturb_noise_std=30.0 --dist_contour_range=4.0 --num_epochs=100 --render_pred=class --save_output
python toy2d_train.py --job_desc=cluster_semisup_run${run} --dataset=img:data/toy2d/curve_mask_v3.png --sup_path=data/toy2d/curve_mask_v3_35.pkl --region_erode_radius=35 --num_epochs=100 --save_output
python toy2d_train.py --job_desc=cluster_sup_run${run} --dataset=img:data/toy2d/curve_mask_v3.png --sup_path=data/toy2d/curve_mask_v3_35.pkl --region_erode_radius=35 --num_epochs=25 --cons_weight=0.0 --save_output
