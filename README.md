# DART: Deformable Anatomy-aware Registration Toolkit for lung ct registration with keypoints supervision
Paper: https://ieeexplore.ieee.org/abstract/document/10635326
<p align="center">
    <img src="figs/overview.png"/> <br />
    <em> 
    Figure 1. An overview of the proposed DART.
    </em>
</p>

Spatially aligning two computed tomography (CT) scans of the lung using automated image registration techniques is a challenging task due to the deformable nature of the lung. However, existing deep-learning-based lung CT registration models are trained with no prior knowledge of anatomical understanding. We propose the deformable anatomy-aware registration toolkit (DART), a masked autoencoder (MAE)-based approach, to improve the keypoint-supervised registration of lung CTs. Our method incorporates features from multiple decoders of networks trained to segment anatomical structures, including the lung, ribs, vertebrae, lobes, vessels, and airways, to ensure that the MAE learns relevant features corresponding to the anatomy of the lung. The pretrained weights of the transformer encoder and patch embeddings are then used as the initialization for the training of downstream registration. We compare DART to existing state-of the-art registration models. Our experiments show that DART outperforms the baseline models (Voxelmorph, ViT-V-Net, and MAE-TransRNet) in terms of target registration error of both corrField-generated keypoints with 17%, 13%, and 9% relative improvement, respectively, and bounding box centers of nodules with 27%, 10%, and 4% relative improvement, respectively.

## Getting Started
```bash
git clone https://github.com/yunzhengzhu/DART.git
cd DART
```
### Pretrained Weights 
Please download from [link](https://drive.google.com/drive/folders/1YVV1BAR6xSVu07Hsqdpw7rLioRqvvOzx?usp=drive_link) and save the weights folder under your `DART` folder.


### Pre-requisities:
- NVIDIA GPU
- python 3.8.12
- pytorch 1.11.0a0+b6df043
- numpy 1.24.2
- pandas 1.3.4
- nibabel 5.1.0
- scipy 1.9.1
- natsort 8.4.0
- einops 0.7.0
- matplotlib 3.5.0
- tensorboardX 2.6.2.2
- scikit-learn 0.24.0
- torchio 0.19.6

### Setup Environment
Install pytorch and python by the following options for environment setup
#### Option 1: Docker (Recommended)
```bash
docker run --shm-size=2g --gpus all -it --rm -v .:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:21.12-py3
```
#### Option 2: Conda
```bash
conda create -n dart python=3.8.12
conda activate dart
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

You can install other dependencies by
```bash
pip install -r requirements.txt
``` 

### Dataset

Download NLST from [Learn2Reg](https://learn2reg.grand-challenge.org/Datasets/) Datasets. The entire dataset consists of 210 pairs of lung CT scans (fixed as baseline and moving as followup) is already preprocessed to a fixed sized 224 x 192 x 224 with spacing 1.5 x 1.5 x 1.5. The corresponding lung masks and keypoints are also provided. For details, please refer to [Learn2Reg2023](https://learn2reg.grand-challenge.org/learn2reg-2023/).
```bash
mkdir -p DATA
cd DATA
wget https://cloud.imi.uni-luebeck.de/s/pERQBNyEFNLY8gR/download/NLST2023.zip
```

#### Unzip the folder under your `DATA` folder
```bash
unzip NLST2023.zip
```
### Segmentation
Please refer to [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for generating the masks (lung, lung lobes, pulmonary vessels, airways, vertebrates, ribs, etc.)

**Note: Please flipping the image to RAS orientation before using totalsegmentator, and flipping the segmentation back to the same orientation as the registration dataset.**

### Nodule Center Generation

Please refer to [MONAI lung nodule detection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/monaitoolkit/models/monai_lung_nodule_ct_detection) for generating the stats (box coordinates and probabilities) for lung nodule bounding boxes. The centers of the bounding box are used to evaluate the registration as the `TRE_nodule` metric.


## Running Experiments
### Training with baselines require no pretraining ([Voxelmorph](https://github.com/voxelmorph/voxelmorph), [ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch))
#### Downstream Registration
```bash
CUDA_VISIBLE_DEVICES='0' python main.py --data_dir DATA/NLST --json_file NLST_dataset.json --result_dir exp --exp_name test --mind_feature --preprocess --use_scaler --downsample 2 --model_type 'Vxm' --loss 'TRE' --loss_weight 1.0 --diff --opt 'adam' --lr 1e-4 --sche 'lambdacosine' --max_epoch 300.0 --lrf 0.01 --batch_size 1 --epochs 300 --seed 1234 --es --es_warmup 0 --es_patience 300 --es_criterion 'TRE' --log --print_every 10
```

**Note: For `model_type` in downstream registration, you need to modify the argument for different models (`Vxm` for Voxelmorph, `ViT-V-Net` for ViT-V-Net, `MAE-ViT_Baseline` for MAE-TransRNet, `MAE_ViT_Seg` for DART)**

**Note: You could skip this step by using our trained weights from `weights/registration/vxm/es_checkpoint.pth.tar` or `weights/registration/vitvnet/es_checkpoint.pth.tar`**

#### Evaluation
```bash
exp_dir=weights/registration/vxm
CUDA_VISIBLE_DEVICES='0' python eval.py --exp_dir ${exp_dir} --save_df --save_warped --eval_diff --mode val
```
A `results_val.csv` will be generated under your `${exp_dir}` folder.

### Training with baselines require pretraining ([MAE-TransRNet](https://github.com/XinXiao101/MAE-TransRNet/tree/main))
#### Pretraining
```bash
CUDA_VISIBLE_DEVICES='0' python pretrain_baseline.py --data_dir DATA/NLST --json_file NLST_dataset.json --result_dir exp --exp_name pt_test --preprocess --use_scaler --mind_feature --downsample 8 --model_type 'MAE_ViT' --loss 'MSE' --loss_weight 1.0 --opt 'adam' --lr 1e-4 --sche 'lambdacosine' --max_epoch 300.0 --lrf 0.01 --batch_size 1 --epochs 1 --seed 1234 --es --es_warmup 0 --es_patience 300 --es_criterion 'MSE' --log --print_every 10
```

**Note: You could skip this step by using our trained weights from `weights/pretraining/mae_t.pth.tar`**

#### Downstream Registration
Same script as `Downstream Registration` above, but remember to load the pretrained weights by adding the argument `--pretrained ${pretrained_weights}`. Remember to specify `model_type` as `MAE_ViT_Baseline` before running.

**Note: You could skip this step by using our trained weights from `weights/registration/mae_t/es_checkpoint.pth.tar`**

#### Evaluation
```bash
exp_dir=weights/registration/mae_t
CUDA_VISIBLE_DEVICES='0' python eval.py --exp_dir ${exp_dir} --save_df --save_warped --eval_diff --mode val
```
A `results_val.csv` will be generated under your `${exp_dir}` folder.

### Our proposed method: DART
**Note: Please prepare segmentation masks (Lung, Lung Lobes, Airways, Pulmonary Vessels, etc.) before doing the following steps.** 

#### Pretraining
```bash
mask_dir=DATA/TOTALSEG_MASK
CUDA_VISIBLE_DEVICES='0' python pretrain_baseline.py --data_dir DATA/NLST --json_file NLST_dataset.json --result_dir exp --exp_name pt_test --preprocess --use_scaler --mind_feature --downsample 8 --model_type 'MAE_ViT_Seg' --loss 'MSE' 'Seg_MSE' --loss_weight 1.0 1.0 --opt 'adam' --lr 1e-4 --sche 'lambdacosine' --max_epoch 300.0 --lrf 0.01 --batch_size 1 --epochs 1 --seed 1234 --es --es_warmup 0 --es_patience 300 --es_criterion 'MSE' --log --print_every 10 --mask_dir ${mask_dir} --eval_with_mask --specific_regions 'lung_trachea_bronchia'
```
**Note: Arguments particularly designed for DART: (examples are using filenames from TotalSegmentator)**
`mask_dir`: specify the dir saving the generated masks
`eval_with_mask`: using the generated masks or not
`specific_regions`: specific regions used for anatomy-aware pretraining
- airways (A): `'lung_trachea_bronchia'`
- vessels (Ve.): `'lung_vessels'`
- 5 lobes (Lo.): `'lung_lower_lobe_left' 'lung_lower_lobe_right' 'lung_middle_lobe_right' 'lung_upper_lobe_left' 'lung_upper_lobe_right'`

`organs`: specify the organ (files contain this organ\'s name)
- lung + rib (LR): `'lung' 'rib'`
- lung + vertebrae (LV): `'lung' 'vertebrae'`
- lung + rib + vertebrae (LRV): `'lung' 'rib' 'vertebrae'`

**Note: You could skip this step by using our trained weights from `weights/pretraining/dart_airways.pth.tar`**

#### Downstream Registration
Same script as `Downstream Registration` above, but remember to load the pretrained weights by adding the argument `--pretrained weights/dart_airways.pth.tar` or your own trained weights. Remember to specify `model_type` as `MAE_ViT_Baseline` before running.

**Note: You could skip this step by using our trained weights from `weights/registration/dart_airways/es_checkpoint.pth.tar`**

#### Evaluation
```bash
exp_dir=weights/registration/dart_airways
CUDA_VISIBLE_DEVICES='0' python eval.py --exp_dir ${exp_dir} --save_df --save_warped --eval_diff --mode val
```
A `results_val.csv` will be generated under your `${exp_dir}` folder.

### Evaluation Metrics
`TRE_kp`: Target registration error of [corrField](https://github.com/multimodallearning/Lung250M-4B/tree/main/corrfield)-generated keypoints <br/>
`TRE_nodule`: Target registration error of bounding box centers generated from MONAI nodule detection algorithm (Please see Nodule Center Generation section above) <br/>
`SDLogJ`: std of the logarithm of the Jacobian determinant of the displacement vector field <br/>


## Citation
```bibtex
@INPROCEEDINGS{10635326,
  author={Zhu, Yunzheng and Zhuang, Luoting and Lin, Yannan and Zhang, Tengyue and Tabatabaei, Hossein and Aberle, Denise R and Prosper, Ashley E and Chien, Aichi and Hsu, William},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)}, 
  title={DART: Deformable Anatomy-Aware Registration Toolkit for Lung CT Registration with Keypoints Supervision}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Image registration;Computed tomography;Computational modeling;Lung;Brain modeling;Transformers;Lung image registration;Masked Autoencoder;Anatomy-Aware Pretraining},
  doi={10.1109/ISBI56570.2024.10635326}}
```
