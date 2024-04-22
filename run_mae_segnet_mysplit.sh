#!/usr/bin/env bash
export MAIN_ROOT= # LEARN2REG ROOT
export EVAL_ROOT= # EVAL ROOT
export PATH=$PWD:$MAIN_ROOT:$MAIN_ROOT/dataset:$MAIN_ROOT/model:$MAIN_ROOT/utils:$EVAL_ROOT:$PATH

BASE_PATH= # Data Path
json_file= # Data Json File
stage=$1
stop_stage=$2

if [ $stage -le 0 ] && [ $stop_stage -ge -1 ]; then
	if [ $# -eq 3 ]; then
		sub_dir=$3
		if [[ ${sub_dir} == *"keypoints"* ]]; then
			kp_dir=${sub_dir}
		elif [[ ${sub_dir} == *"masks"* ]]; then
			mask_dir=${sub_dir}
		fi
	elif [ $# -eq 4 ]; then
		sub_dir1=$3
		sub_dir2=$4
		if [[ ${sub_dir1} == *"keypoints"* ]]; then
			kp_dir=${sub_dir1}
		elif [[ ${sub_dir1} == *"masks"* ]]; then
			mask_dir=${sub_dir1}
		fi
		if [[ ${sub_dir2} == *"keypoints"* ]]; then
			kp_dir=${sub_dir2}
		elif [[ ${sub_dir1} == *"masks"* ]]; then
			mask_dir=${sub_dir2}
		fi
	fi
elif [ $stage -ge 1 ]; then
        exp_dir=$3
fi


exp=exp_nearest_mysplit_pt
exp_name='test'
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
        echo 'Pretrain Start'
        CUDA_VISIBLE_DEVICES='0' pretrain_segnet.py \
		--data_dir ${BASE_PATH} \
                --json_file ${json_file} \
                --result_dir ${exp} \
                --exp_name ${exp_name} \
		--mind_feature \
                --preprocess \
                --use_scaler \
		--downsample 8 \
		--model_type 'MAE_ViT_Seg' \
		--loss 'MSE' 'Seg_MSE' \
                --loss_weight 1.0 1.0 \
                --opt 'adam' \
                --lr 1e-4 \
                --sche 'lambdacosine' \
		--max_epoch 300.0 \
		--lrf 0.01 \
                --batch_size 1 \
                --epochs 1 \
                --seed 1234 \
                --es \
                --es_warmup 0 \
                --es_patience 300 \
                --es_criterion 'MSE' \
                --log \
		--mask_dir ${mask_dir} \
		--eval_with_mask \
		--specific_regions 'lung_lower_lobe_left' 'lung_lower_lobe_right' 'lung_middle_lobe_right' 'lung_upper_lobe_left' 'lung_upper_lobe_right' 'lung_trachea_bronchia' 'lung_vessels' \
                --print_every 10 #> train.log
		#--organs 'lobe' 'trachea_bronchia' 'vessels' \
fi

exp=exp_nearest_mysplit_ft
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
        echo 'Experiment Start'
        CUDA_VISIBLE_DEVICES='3' main.py \
		--pretrained exp_nearest_mysplit_pt/test/checkpoint.pth.tar \
		--data_dir ${BASE_PATH} \
                --json_file ${json_file} \
                --result_dir ${exp} \
                --exp_name ${exp_name} \
		--mind_feature \
                --preprocess \
                --use_scaler \
		--downsample 2 \
		--model_type 'MAE_ViT_Baseline' \
		--loss 'TRE' \
                --loss_weight 1.0 \
		--diff \
                --opt 'adam' \
                --lr 1e-4 \
                --sche 'lambdacosine' \
		--max_epoch 300.0 \
		--lrf 0.01 \
                --batch_size 1 \
                --epochs 1 \
                --seed 1234 \
                --es \
                --es_warmup 0 \
                --es_patience 300 \
                --es_criterion 'TRE' \
                --log \
                --print_every 10 #> train.log
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
        echo 'Evaluation Start'
 	for part in test; do
		CUDA_VISIBLE_DEVICES='0' eval.py \
			--exp_dir ${exp_dir} \
			--save_df \
			--save_warped \
			--eval_diff \
			--mode ${part}
	done
			#--nodule_kp_dir ${nodule_kp_dir} \
			#--nodule_id ${nodule_file} \
fi
