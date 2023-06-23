#!/usr/bin/env bash

export MAIN_ROOT= # LEARN2REG ROOT
export PATH=$PWD:$MAIN_ROOT:$MAIN_ROOT/dataset:$MAIN_ROOT/model:$MAIN_ROOT/utils:$PATH

BASE_PATH=/workspace/imgregdata/NLST2023
json_file=NLST_dataset_test.json
stage=$1
stop_stage=$2
if [ $stop_stage -ge 1 ]; then
	exp_dir=$3
fi

exp=exp
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
	echo 'Training Start'
	CUDA_VISIBLE_DEVICES='2' main.py \
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
		--sche 'lambdacosine' \
		--max_epoch 100 \
		--lrf 0.1 \
		--es \
		--es_warmup 0 \
		--es_patience 20 \
		--log \
		--print_every 10
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
	echo 'Evaluation Start'
	CUDA_VISIBLE_DEVICES='2' eval.py \
		--exp_dir ${exp_dir} \
		--save_df
fi
