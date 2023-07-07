# learn2reg2023


### Deep Learning Based Models 
#### Command line

**For training**
```bash
CUDA_VISIBLE_DEVICES='1' main.py \
	--data_dir {DATA_DIR} \
	--json_file {JSON_FILE} \
	--result_dir {RESULT_DIR} \
	--model_type 'LKU-Net' \
	--start_channel 8 \
	--loss 'NCC' 'Smooth' \
	--loss_weight 1.0 0.25 \
	--opt 'adam' \
	--lr 1e-3 \
	--batch_size 1 \
	--epochs 5 \
	--seed 1234 \
	--sche 'lambdacosine' \
	--max_epoch 100 \
	--lrf 0.1 \
	--rev_metric \
	--es \
	--es_warmup 0 \
	--es_patience 20 \
	--log \
	--print_every 20
```
Note: `sche`, `max_epoch`, `lrf` are optional. 

**If your training got interupted**
```bash
CUDA_VISIBLE_DEVICES='1' main.py \
	--continue_training \
	--exp_dir {EXPERIMENT_DIR} \
	--epochs 10 #total num epochs (required - otherwise it will be default 100)
```

**For inference**
```bash
CUDA_VISIBLE_DEVICES='1' eval.py \
	--exp_dir {EXPERIMENT_DIR} \
	--save_df
```

#### TODO

- DataLoader
  - [x] NLST dataloader
  - [ ] ThoraxCBCT dataloader
- Models
  - [ ] resnet + unet (baseline)
  - [ ] unet (baseline architecture)
  - [ ] lkunet (baseline architecture)
  - [x] lkunet 
  - [x] voxelmorph (unet)
  - [ ] transmorph
- Loss Functions
  - [x] NCC
  - [x] GNCC
  - [x] MSE
  - [x] Dice
  - [x] SAD
  - [ ] Structural similarity index (SSIM)
  - [ ] Mutual information (MI)
  - [ ] Local mutual information (LMI)
  - [ ] Modality independent neighbourhood descriptor with self-similarity context (MIND-SSC)
  - [x] MSE for keypoints
  - [ ] ...
- Metrics
  - [x] TRE(keypoints)
  - [x] Dice(masks)
  - [x] Jacobian determinant
  - [ ] ...

- Others
  - [ ] save as nifti (original image size, real displacement)
  - [x] add scheduler (cosine, lambdacosine)
  - [ ] train val split
  - [ ] early stopping metric
