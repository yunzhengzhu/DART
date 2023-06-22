#!/usr/bin/env bash

export MAIN_ROOT= # LEARN2REG ROOT
export PATH=$PWD:$MAIN_ROOT:$MAIN_ROOT/dataset:$MAIN_ROOT/model:$MAIN_ROOT/utils:$PATH

BASE_PATH=/workspace/imgregdata/NLST2023
json_file=NLST_dataset_test.json
stage=0
stop_stage=1

exp=exp
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
	echo 'Training Start'
	CUDA_VISIBLE_DEVICES='1' main.py \
		--data_dir ${BASE_PATH} \
		--json_file ${json_file} \
		--result_dir ${exp} \
		--model_type 'LKU-Net' \
		--start_channel 8 \
		--loss 'NCC' 'Smooth' \
		--loss_weight 1.0 0.25 \
		--opt 'adam' \
		--lr 1e-3 \
		--batch_size 1 \
		--epochs 1 \
		--seed 1234 \
		--es \
		--es_warmup 10 \
		--es_tolerence 20 \
		--log \
		--print_every 10
fi

exp_dir=${exp}/LKU-Net_NCC_Smooth_adam_lr0.001_bs1_seed1234/
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
	echo 'Evaluation Start'
	CUDA_VISIBLE_DEVICES='1' eval.py \
		--exp_dir ${exp_dir} \
		--save_df
fi
