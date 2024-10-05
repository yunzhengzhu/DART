import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import os
import json
import numpy as np
import scipy
import copy
from typing import Tuple
from natsort import natsorted
from utils.feature_utils import foerstner_kpts, knn_match

class NLSTDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        json_file: str, 
        mode: str = "train", 
        downsample: int = 1, 
        preprocess: bool = False,
        orient_stand: bool = False,
        random_sample: int = None, 
        kp_dir: str = None,
        nodule_kp_dir: str = None,
        nodule_id: str = None,
        lm: str = None,
        mask_dir: str = None,
        mask_info: dict = {},
        affine_aug: str = None,
        affine_prob: float = 0.5,
        affine_param: float = 0.035,
        flip_aug: str = None,
        flip_prob: float = 0.5,
        flip_axis: list = [1, -1],
        kp_aug: bool = False,
        kp_aug_info: dict = {},
        eval_with_mask: bool = False,
        texture_mask_dir: str = None,
    ) -> None:
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode
        self.downsample = downsample
        self.preprocess = preprocess
        self.orient_stand = orient_stand
        self.random_sample = random_sample
        self.kp_dir = kp_dir
        self.nodule_kp_dir = nodule_kp_dir
        self.nodule_id = nodule_id
        self.lm = lm
        self.mask_dir = mask_dir
        self.eval_with_mask = eval_with_mask
        if self.mask_dir or self.eval_with_mask:
            self.organs = mask_info["organs"]
            self.side = mask_info["side"]
            self.specific_regions = mask_info["specific_regions"]
        else:
            self.organs = None
            self.side = None
            self.specific_regions = None
        self.texture_mask_dir = texture_mask_dir
        self.affine_aug = affine_aug
        self.affine_prob = affine_prob
        self.affine_param = affine_param
        self.flip_aug = flip_aug
        self.flip_prob = flip_prob
        self.flip_axis = flip_axis
        self.kp_aug = kp_aug
        if self.kp_aug:
            self.kernel = kp_aug_info["kernel"] #args.foerstner_kernel
            self.N_P = kp_aug_info["num_points"] #args.foerstner_points
            self.T = kp_aug_info["threshold"] #args.points_thres

        # read json file
        with open(os.path.join(data_dir, json_file)) as jf:
            file_contents = jf.read()
            jdict = json.loads(file_contents)

        self.H = jdict["tensorImageShape"]["0"][0]
        self.W = jdict["tensorImageShape"]["0"][1]
        self.D = jdict["tensorImageShape"]["0"][2]

        # get subjects based on mode
        if mode == "train":
            subjects = jdict["training_paired_images"]
            # remove duplicate cases from val
            subjects = [subject for subject in subjects if (subject not in jdict["registration_val"]) and (subject not in jdict["registration_test"])]
            # remove 3 pairs that are not in the right orientation
            #subjects = [subject for subject in subjects if ('0208' not in subject['fixed']) and ('0206' not in subject['fixed']) and ('0298' not in subject['fixed'])]
            
            if self.side == ["both"]:
                subjects1 = copy.deepcopy(subjects)
                for i in range(len(subjects1)):
                    subjects1[i]["side"] = ["left"]

                subjects2 = copy.deepcopy(subjects)
                for i in range(len(subjects2)):
                    subjects2[i]["side"] = ["right"]
               
                subjects = subjects1 + subjects2

        elif mode == "val":
            subjects = jdict["registration_val"]
        elif mode == "test":
            subjects = jdict["registration_test"]
        
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        # load fixed and moving images
        # image size (224,192,224) -> (1,224,192,224)
        fixed_relative_path = os.path.normpath(self.subjects[idx]["fixed"])
        moving_relative_path = os.path.normpath(self.subjects[idx]["moving"])
        file_id = fixed_relative_path.split("_")[1]

        fixed_img_path = os.path.join(self.data_dir, fixed_relative_path)
        moving_img_path = os.path.join(self.data_dir, moving_relative_path)

        fixed_img = self.__load_nii_img(fixed_img_path, preprocess=self.preprocess, downsample=self.downsample)[None, ...]
        moving_img = self.__load_nii_img(moving_img_path, preprocess=self.preprocess, downsample=self.downsample)[None, ...]
        
        # load fixed and moving keypoints
        #if self.mode == "test":
        #    fixed_kp = np.zeros((1, 3))[None, ...]
        #    moving_kp = np.zeros((1, 3))[None, ...]
        #elif self.mode == "train" and self.kp_dir:
        if self.mode == "train" and self.kp_dir:
            fixed_kp = np.genfromtxt(
                            fixed_img_path.replace("images", "keypoints").replace("nii.gz", "csv").replace("keypointsTr", self.kp_dir),
                            delimiter=",",
                        )[None, ...]
            moving_kp = np.genfromtxt(
                            moving_img_path.replace("images", "keypoints").replace("nii.gz", "csv").replace("keypointsTr", self.kp_dir),
                            delimiter=",",
                        )[None, ...]
        else:
            fixed_kp = np.genfromtxt(
                            fixed_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
                            delimiter=",",
                       )[None, ...]
            
            moving_kp = np.genfromtxt(
                            moving_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
                            delimiter=",",
                        )[None, ...]

        ### nodule keypoints #####################################################################################
        if (self.mode == "val" or self.mode == "test") and self.nodule_kp_dir:
            if self.nodule_id == "all":
                fixed_nodu_kp_dir = os.path.splitext(os.path.splitext(fixed_img_path)[0])[0].replace("imagesTr", self.nodule_kp_dir)
                moving_nodu_kp_dir = os.path.splitext(os.path.splitext(moving_img_path)[0])[0].replace("imagesTr", self.nodule_kp_dir)
                fixed_nodu_kp = np.zeros((0, 3))
                moving_nodu_kp = np.zeros((0, 3))
                if self.lm == "nodule_kpt":
                    for fl in os.listdir(fixed_nodu_kp_dir):
                        if "forward" in fl:
                            this_fixed_nodu_kp = np.genfromtxt(os.path.join(fixed_nodu_kp_dir, fl), delimiter=",")
                            fixed_nodu_kp = np.concatenate((fixed_nodu_kp, this_fixed_nodu_kp), axis=0)
                            this_moving_nodu_kp = np.genfromtxt(os.path.join(moving_nodu_kp_dir, fl), delimiter=",")
                            moving_nodu_kp = np.concatenate((moving_nodu_kp, this_moving_nodu_kp), axis=0)
                elif self.lm == "nodule_center":
                    if "boxes.json" in os.listdir(fixed_nodu_kp_dir):
                        fixed_boxes = json.loads(open(os.path.join(fixed_nodu_kp_dir, "boxes.json")).read())["box"]
                        moving_boxes = json.loads(open(os.path.join(moving_nodu_kp_dir, "boxes.json")).read())["box"]
                        fixed_boxes_score = json.loads(open(os.path.join(fixed_nodu_kp_dir, "boxes.json")).read())["score"]
                        moving_boxes_score = json.loads(open(os.path.join(moving_nodu_kp_dir, "boxes.json")).read())["score"]
                        for bid in range(len(fixed_boxes)):
                            if fixed_boxes_score[bid] >= 0.8:
                                this_fixed_nodu_kp = np.array([
                                    (fixed_boxes[bid][0] + fixed_boxes[bid][0+3])/2,
                                    (fixed_boxes[bid][1] + fixed_boxes[bid][1+3])/2,
                                    (fixed_boxes[bid][2] + fixed_boxes[bid][2+3])/2
                                ])[None, ...]
                                fixed_nodu_kp = np.concatenate((fixed_nodu_kp, this_fixed_nodu_kp), axis=0)
                                this_moving_nodu_kp = np.array([
                                    (moving_boxes[bid][0] + moving_boxes[bid][0+3])/2,
                                    (moving_boxes[bid][1] + moving_boxes[bid][1+3])/2,
                                    (moving_boxes[bid][2] + moving_boxes[bid][2+3])/2
                                ])[None, ...]
                                moving_nodu_kp = np.concatenate((moving_nodu_kp, this_moving_nodu_kp), axis=0)
                
                fixed_nodu_kp = fixed_nodu_kp[None, ...]
                moving_nodu_kp = moving_nodu_kp[None, ...]
            else:
                fixed_nodu_kp_path = os.path.join(os.path.splitext(os.path.splitext(fixed_img_path)[0])[0].replace("imagesTr", self.nodule_kp_dir), f"{self.nodule_id}")
                moving_nodu_kp_path = os.path.join(os.path.splitext(os.path.splitext(moving_img_path)[0])[0].replace("imagesTr", self.nodule_kp_dir), f"{self.nodule_id}")
                if os.path.exists(fixed_nodu_kp_path) and os.path.exists(moving_nodu_kp_path):
                    
                    fixed_nodu_kp = np.genfromtxt(fixed_nodu_kp_path, delimiter=",")[None, ...]
                    moving_nodu_kp = np.genfromtxt(moving_nodu_kp_path, delimiter=",")[None, ...]
                else:
                    if not os.path.exists(fixed_nodu_kp_path):
                        print(f"nodule file {fixed_nodu_kp_path} does not exist!")
                    if not os.path.exists(fixed_nodu_kp_path):
                        print(f"nodule file {moving_nodu_kp_path} does not exist!")
                    fixed_nodu_kp = np.zeros((0, 3))[None, ...]
                    moving_nodu_kp = np.zeros((0, 3))[None, ...]
        #############################################################################################################


        if self.texture_mask_dir:
            # texture mask
            fixed_texture_mask = self.__load_nii_img(
                fixed_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
            moving_texture_mask = self.__load_nii_img(
                moving_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
        
        # load masks
        if (self.mode == "train" and self.mask_dir) or ((self.mode == "val" or self.mode == "test") and self.eval_with_mask):
            #print(f"==== load {self.mode} masks from {self.mask_dir}")
            if self.side == ["both"]:
                side = self.subjects[idx]["side"]
            else:
                side = self.side
            #print(f"original: {fixed_kp.shape} {moving_kp.shape}")
            fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, labels = self.__load_mask_dir(fixed_img_path, moving_img_path, side=side)
            #print(fixed_multiple_mask.shape)
            #print(moving_multiple_mask.shape)

            if self.texture_mask_dir:
                fixed_mask = fixed_texture_mask * fixed_mask
                moving_mask = moving_texture_mask * moving_mask
            #print(f"original: {fixed_kp.max()} {moving_kp.max()}")
            fixed_kp, moving_kp = self.__filter_mask_based_kpt(
                fixed_kp.squeeze(0), 
                moving_kp.squeeze(0), 
                fixed_mask.squeeze(0), 
                moving_mask.squeeze(0)
            )
            fixed_kp = fixed_kp[None, ...]
            moving_kp = moving_kp[None, ...]
            #print(f"filter: {fixed_kp.shape} {moving_kp.shape}")
            #print(f"filter: {fixed_kp.max()} {moving_kp.max()}")
        else:
            fixed_mask = self.__load_nii_img(
                fixed_img_path.replace("images", "masks"), preprocess=False, downsample=self.downsample
            )[None, ...]
            moving_mask = self.__load_nii_img(
                moving_img_path.replace("images", "masks"), preprocess=False, downsample=self.downsample
            )[None, ...]
            if self.texture_mask_dir:
                fixed_mask = fixed_texture_mask * fixed_mask
                moving_mask = moving_texture_mask * moving_mask
            fixed_multiple_mask = fixed_mask.copy()
            moving_multiple_mask = moving_mask.copy()
            labels = ['lung']
       
        # mask
        #fixed_img[~fixed_mask.astype(bool)] = fixed_img.min()
        #moving_img[~moving_mask.astype(bool)] = moving_img.min()

        # standarize the orientation
        # only need for NLST data (by visualizing the data, 0208, 0260, and 0298 cases have different orientations, potentially not helpful for this validation data without flipped orientation)
        if self.orient_stand:
            flip_axis=None
            if '0208' in file_id:
                flip_axis = [-1]
            elif '0260' in file_id:
                flip_axis = [1]
            elif '0298' in file_id:
                flip_axis = [-1]
            if flip_axis != None:
                (
                    fixed_img,
                    fixed_mask,
                    fixed_multiple_mask,
                    fixed_kp,
                ) = self.__standardize_orientation(
                    fixed_img[0],
                    fixed_mask[0],
                    fixed_multiple_mask,
                    fixed_kp[0],
                    flip_axis=flip_axis,
                )
                
                (
                    moving_img,
                    moving_mask,
                    moving_multiple_mask,
                    moving_kp,
                ) = self.__standardize_orientation(
                    moving_img[0],
                    moving_mask[0],
                    moving_multiple_mask,
                    moving_kp[0],
                    flip_axis=flip_axis,
                )
            #print(fixed_multiple_mask.shape)
            #print(moving_multiple_mask.shape)

        if self.mode == 'train':
            # randomly sample keypoints
            if self.random_sample != None:
                random_sample_kp = np.random.choice(fixed_kp.shape[1], self.random_sample, replace=False)
                fixed_kp = fixed_kp[:, random_sample_kp, :]
                moving_kp = moving_kp[:, random_sample_kp, :]

            # affine transform
            if self.affine_aug:
                # prob of affine transform
                if self.affine_aug == "moving" or self.affine_aug == "both":
                    moving_aff_prob = np.random.uniform(0, 1)
                else:
                    moving_aff_prob = 0.0

                if self.affine_aug == "fixed" or self.affine_aug == "both":
                    fixed_aff_prob = np.random.uniform(0, 1)
                else:
                    fixed_aff_prob = 0.0

                A = torch.randn(3, 4) * self.affine_param + torch.eye(3, 4)

                affine = F.affine_grid(A.unsqueeze(0), (1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), align_corners=True)

                # normalize kp
                if fixed_aff_prob > self.affine_prob:
                    (
                        fixed_img, 
                        fixed_mask,
                        fixed_multiple_mask,
                        fixed_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(fixed_img).squeeze(0).float(), 
                        torch.from_numpy(fixed_mask).squeeze(0).float(),
                        torch.from_numpy(fixed_multiple_mask).float(),
                        torch.from_numpy(fixed_kp).squeeze(0).float(),
                        affine,
                        A
                    )
                if moving_aff_prob > self.affine_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_multiple_mask,
                        moving_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(moving_img).squeeze(0).float(), 
                        torch.from_numpy(moving_mask).squeeze(0).float(), 
                        torch.from_numpy(moving_multiple_mask).float(),
                        torch.from_numpy(moving_kp).squeeze(0).float(), 
                        affine,
                        A
                    )
            
            # flip transform
            if self.flip_aug:
                # prob of flip transform
                prob = np.random.uniform(0, 1)
                if self.flip_aug == "moving" or self.flip_aug == "both":
                    moving_flip_prob = prob
                else:
                    moving_flip_prob = 0.0

                if self.flip_aug == "fixed" or self.flip_aug == "both":
                    fixed_flip_prob = prob
                else:
                    fixed_flip_prob = 0.0

                flip_axis = self.flip_axis

                # normalize kp
                if fixed_flip_prob > self.flip_prob:
                    (
                        fixed_img, 
                        fixed_mask,
                        fixed_multiple_mask,
                        fixed_kp
                    ) = self.__apply_flip_augmentation(
                        fixed_img[0], 
                        fixed_mask[0], 
                        fixed_multiple_mask,
                        fixed_kp[0],
                        flip_axis
                    )
                if moving_flip_prob > self.flip_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_multiple_mask,
                        moving_kp
                    ) = self.__apply_flip_augmentation(
                        moving_img[0], 
                        moving_mask[0], 
                        moving_multiple_mask,
                        moving_kp[0], 
                        flip_axis
                    )
            
            if self.kp_aug:
                _, fixed_kp_aug = foerstner_kpts(
                    torch.from_numpy(fixed_img)[None, ...].to(torch.float32), 
                    torch.from_numpy(fixed_mask)[None, ...].to(torch.float32), 
                    sigma=1.4,
                    d=self.kernel,
                    thresh=1e-8,
                    num_points=self.N_P
                )
                _, moving_kp_aug = foerstner_kpts(
                    torch.from_numpy(moving_img)[None, ...].to(torch.float32),
                    torch.from_numpy(moving_mask)[None, ...].to(torch.float32),
                    sigma=1.4,
                    d=self.kernel,
                    thresh=1e-8,
                    num_points=self.N_P
                )
                print(fixed_kp_aug.shape, moving_kp_aug.shape)

                (
                    fixed_kp_aug_match, 
                    moving_kp_aug_match
                ) = knn_match(
                    fixed_kp_aug, 
                    moving_kp_aug,
                    k=1,
                    T=self.T,
                )
                print(fixed_kp_aug_match.shape, moving_kp_aug_match.shape)

                fixed_kp = np.concatenate((fixed_kp, fixed_kp_aug_match), axis=1)
                moving_kp = np.concatenate((moving_kp, moving_kp_aug_match), axis=1)
                
                print(fixed_kp.shape, moving_kp.shape)
   
        if self.nodule_kp_dir:
            return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, fixed_nodu_kp, moving_nodu_kp, labels 
        else:
            return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, labels

    def __standardize_orientation(self, img, mask, multiple_mask, kp, flip_axis):
        flip_kp = kp.copy() 
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            so_multiple_mask = np.zeros_like(multiple_mask)
            for i in range(multiple_mask.shape[0]):
                so_multiple_mask[i:i+1] = np.flip(multiple_mask[i], axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, so_multiple_mask, flip_kp
    
    def __apply_flip_augmentation(self, img, mask, multiple_mask, kp, flip_axis):
        flip_kp = kp.copy()
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_multiple_mask = np.zeros_like(multiple_mask)
            for i in range(multiple_mask.shape[0]):
                flip_multiple_mask[i:i+1] = np.flip(multiple_mask[i], axis=fa).copy()

            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_multiple_mask, flip_kp

    def __apply_affine_augmentation(self, img, mask, multiple_mask, kp, affine, A):
        # keypoint
        kp = (kp / torch.tensor([self.H // self.downsample, self.W // self.downsample, self.D // self.downsample]) * 2 - 1).flip(-1)
        kp = self.__normalize_kp(kp)

        kp = (torch.solve(torch.cat((kp, torch.ones(kp.shape[0], 1)), 1).float().t(),
              torch.cat((A, torch.tensor([0, 0, 0, 1]).view(1, -1)), 0))[0].t()[:, :3].squeeze())

        kp = ((kp + 1) / 2 * torch.tensor([self.H // self.downsample, self.W // self.downsample, self.D // self.downsample])).flip(-1)
        kp = kp.unsqueeze(0).numpy()

        # img
        img = F.grid_sample(img.view(1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        img = img.unsqueeze(0).numpy()

        # mask
        mask = F.grid_sample(mask.view(1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        mask = mask.unsqueeze(0).numpy()

        # multiple mask
        aff_multiple_mask = F.grid_sample(multiple_mask.view(1, multiple_mask.shape[0], self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        aff_multiple_mask = aff_multiple_mask.unsqueeze(0).numpy()
        
        #test_aff_multiple_mask = np.zeros_like(multiple_mask)
        #for i in range(multiple_mask.shape[0]):
        #    test_aff_multiple_mask[i:i+1] = F.grid_sample(multiple_mask[i:i+1].view(1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze().unsqueeze(0).numpy()
        return img, mask, aff_multiple_mask, kp
       

    def __load_mask_dir(self, fixed_img_path, moving_img_path, side=None): 
        fixed_mask_dir = fixed_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        moving_mask_dir = moving_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        
        # extract and check the available masks in nifti
        all_nifti_f = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])
        all_nifti_m = natsorted([i.split(".")[0] for i in os.listdir(moving_mask_dir) if "nii" in i])
        assert all_nifti_f == all_nifti_m
        all_nifti = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])

        # filter out masks based on requirements (organ, side, specific lobe)
        if self.organs or side or self.specific_regions:
            if self.specific_regions:
                selected_nifti = [f for f in all_nifti if any([sr in f for sr in self.specific_regions])]
            else:
                selected_nifti = [f for f in all_nifti if any([organ in f for organ in self.organs])]
                if side:
                    selected_nifti = [f for f in selected_nifti if any([s in f for s in side])]
                   
            all_nifti = selected_nifti
        
        # combine to be one mask for usage
        fixed_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        moving_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        if self.organs:
            mask_class = len(self.organs)
        else:
            mask_class = len(all_nifti)
        fixed_multiple_mask = np.zeros((mask_class, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        moving_multiple_mask = np.zeros((mask_class, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))

        for i, nif in enumerate(all_nifti):
            this_nif_fixed_mask = self.__load_nii_img(
                os.path.join(fixed_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            fixed_mask += this_nif_fixed_mask
            
            this_nif_moving_mask = self.__load_nii_img(
                os.path.join(moving_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            moving_mask += this_nif_moving_mask
           
            if self.organs:
                for j, organ in enumerate(self.organs):
                    if organ in nif:
                        fixed_multiple_mask[j:j+1] += this_nif_fixed_mask
                        moving_multiple_mask[j:j+1] += this_nif_moving_mask
            else:
                fixed_multiple_mask[i:i+1] += this_nif_fixed_mask
                moving_multiple_mask[i:i+1] += this_nif_moving_mask
           
        label_list = all_nifti if not self.organs else self.organs
        return fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, label_list
    
    @staticmethod
    def __filter_mask_based_kpt(fixed_kpt, moving_kpt, fixed_mask, moving_mask, label=1):
        fixed_kpt = fixed_kpt / 2
        H, W, D = fixed_mask.shape
        fixed_kpt_idx = np.round(fixed_kpt).astype(np.int16)
        fixed_kpt_idx[:, 0][fixed_kpt_idx[:, 0] >= H] = H-1
        fixed_kpt_idx[:, 1][fixed_kpt_idx[:, 1] >= W] = W-1
        fixed_kpt_idx[:, 2][fixed_kpt_idx[:, 2] >= D] = D-1
        fixed_kpt_mask = fixed_mask[fixed_kpt_idx[:, 0], fixed_kpt_idx[:, 1], fixed_kpt_idx[:, 2]]

        moving_kpt = moving_kpt / 2
        moving_kpt_idx = np.round(moving_kpt).astype(np.int16)
        moving_kpt_idx[:, 0][moving_kpt_idx[:, 0] >= H] = H-1
        moving_kpt_idx[:, 1][moving_kpt_idx[:, 1] >= W] = W-1
        moving_kpt_idx[:, 2][moving_kpt_idx[:, 2] >= D] = D-1
        moving_kpt_mask = moving_mask[moving_kpt_idx[:, 0], moving_kpt_idx[:, 1], moving_kpt_idx[:, 2]]
        
        filter_idx = np.where(fixed_kpt_mask == label)[0][np.in1d(np.where(fixed_kpt_mask == label)[0], np.where(moving_kpt_mask == label)[0])]
        return fixed_kpt[filter_idx]*2, moving_kpt[filter_idx]*2

    @staticmethod
    def __load_nii_img(img_path, preprocess: bool = False, downsample: int = 1) -> np.ndarray:
        img = nib.load(img_path)
        arr = img.get_fdata()

        if preprocess:
            # clip pixel values
            min_bound = -1000.0
            max_bound = 500.0
            arr[arr < min_bound] = min_bound
            arr[arr > max_bound] = max_bound

            # normalize
            arr = (arr - min_bound) / (max_bound - min_bound)

        if downsample != 1:
            arr = scipy.ndimage.zoom(arr, zoom=(1/downsample, 1/downsample, 1/downsample), order=0)

        return arr


class NLSTDataset_MAEPretrain_Baseline(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        json_file: str, 
        mode: str = "train", 
        downsample: int = 1, 
        preprocess: bool = False,
        orient_stand: bool = False,
        random_sample: int = None, 
        kp_dir: str = None,
        mask_dir: str = None,
        mask_info: dict = {},
        affine_aug: str = None,
        affine_prob: float = 0.5,
        affine_param: float = 0.035,
        flip_aug: str = None,
        flip_prob: float = 0.5,
        flip_axis: list = [1, -1],
        kp_aug: bool = False,
        kp_aug_info: dict = {},
        eval_with_mask: bool = False,
        texture_mask_dir: str = None
    ) -> None:
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode
        self.downsample = downsample
        self.preprocess = preprocess
        self.orient_stand = orient_stand
        self.random_sample = random_sample
        self.kp_dir = kp_dir
        self.mask_dir = mask_dir
        self.eval_with_mask = eval_with_mask
        if self.mask_dir or self.eval_with_mask:
            self.organs = mask_info["organs"]
            self.side = mask_info["side"]
            self.specific_regions = mask_info["specific_regions"]
        else:
            self.organs = None
            self.side = None
            self.specific_regions = None
        self.texture_mask_dir = texture_mask_dir
        self.affine_aug = affine_aug
        self.affine_prob = affine_prob
        self.affine_param = affine_param
        self.flip_aug = flip_aug
        self.flip_prob = flip_prob
        self.flip_axis = flip_axis
        self.kp_aug = kp_aug
        if self.kp_aug:
            self.kernel = kp_aug_info["kernel"] #args.foerstner_kernel
            self.N_P = kp_aug_info["num_points"] #args.foerstner_points
            self.T = kp_aug_info["threshold"] #args.points_thres

        # read json file
        with open(os.path.join(data_dir, json_file)) as jf:
            file_contents = jf.read()
            jdict = json.loads(file_contents)

        self.H = jdict["tensorImageShape"]["0"][0]
        self.W = jdict["tensorImageShape"]["0"][1]
        self.D = jdict["tensorImageShape"]["0"][2]

        subjects = jdict["training"]
        #validation_list = ["0101", "0102", "0103", "0104", "0105", "0106", "0107", "0108", "0109", "0110"]
        validation_list = [
            "0040", "0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049",
            "0090", "0091", "0092", "0093", "0094", "0095", "0096", "0097", "0098", "0099",
            "0240", "0241", "0242", "0243", "0244", "0245", "0246", "0247", "0248", "0249",
            "0290", "0291", "0292", "0293", "0294", "0295", "0296", "0297", "0298", "0299",
        ]
        test_list = ["0101", "0102", "0103", "0104", "0105", "0106", "0107", "0108", "0109", "0110"]
        # get subjects based on mode
        if mode == "train":
            # filter out training cases 
            subjects = [subject for subject in subjects if (subject["image"].split("_")[1] not in validation_list) and (subject["image"].split("_")[1] not in test_list)]
            # remove 3 pairs that are not in the right orientation
            #subjects = [subject for subject in subjects if ('0208' not in subject['fixed']) and ('0206' not in subject['fixed']) and ('0298' not in subject['fixed'])]
            
            if self.side == ["both"]:
                subjects1 = copy.deepcopy(subjects)
                for i in range(len(subjects1)):
                    subjects1[i]["side"] = ["left"]

                subjects2 = copy.deepcopy(subjects)
                for i in range(len(subjects2)):
                    subjects2[i]["side"] = ["right"]
               
                subjects = subjects1 + subjects2

        elif mode == "val":
            # filter out validation cases
            subjects = [subject for subject in subjects if subject["image"].split("_")[1] in validation_list]
        elif mode == "test":
            # filter out validation cases
            subjects = [subject for subject in subjects if subject["image"].split("_")[1] in test_list]

        self.subjects = subjects
        
    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int):
        # load image
        # image size (224,192,224) -> (1,224,192,224)
        relative_path = os.path.normpath(self.subjects[idx]["image"])
        file_id = relative_path.split("_")[1]

        img_path = os.path.join(self.data_dir, relative_path)

        img = self.__load_nii_img(img_path, preprocess=self.preprocess, downsample=self.downsample)[None, ...]
        
        # load keypoints
        if self.mode == "train" and self.kp_dir:
            kp = np.genfromtxt(
                        img_path.replace("images", "keypoints").replace("nii.gz", "csv").replace("keypointsTr", self.kp_dir),
                        delimiter=",",
                    )[None, ...]
        else:
            kp = np.genfromtxt(
                        img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
                        delimiter=",",
                   )[None, ...]
            
        if self.texture_mask_dir:
            # texture mask
            texture_mask = self.__load_nii_img(
                img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
        
        # load masks
        if (self.mode == "train" and self.mask_dir) or ((self.mode == "val" or self.mode == "test") and self.eval_with_mask):
            if self.side == ["both"]:
                side = self.subjects[idx]["side"]
            else:
                side = self.side
            
            mask, multiple_mask, labels = self.__load_mask_dir(img_path, side=side)
            #multiple_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
            #mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
            #labels = []
            #print(multiple_mask.shape)
            #print(labels)
            if self.texture_mask_dir:
                mask = texture_mask * mask
            
            kp = self.__filter_mask_based_kpt(
                kp.squeeze(0), 
                mask.squeeze(0), 
            )
            kp = kp[None, ...]
        else:
            mask = self.__load_nii_img(
                img_path.replace("images", "masks"), preprocess=False, downsample=self.downsample
            )[None, ...]
            if self.texture_mask_dir:
                mask = texture_mask * mask
            multiple_mask = mask.copy()
            labels = ['lung']
       
        # mask
        #fixed_img[~fixed_mask.astype(bool)] = fixed_img.min()

        # standarize the orientation
        # only need for NLST data (by visualizing the data, 0208, 0260, and 0298 cases have different orientations, potentially not helpful for this validation data without flipped orientation)
        if self.orient_stand:
            flip_axis=None
            if '0208' in file_id:
                flip_axis = [-1]
            elif '0260' in file_id:
                flip_axis = [1]
            elif '0298' in file_id:
                flip_axis = [-1]
            if flip_axis != None:
                (
                    img,
                    mask,
                    multiple_mask,
                    kp,
                ) = self.__standardize_orientation(
                    img[0],
                    mask[0],
                    multiple_mask,
                    kp[0],
                    flip_axis=flip_axis,
                )
            #print(multiple_mask.shape)
                
        if self.mode == 'train':
            # randomly sample keypoints
            if self.random_sample != None:
                random_sample_kp = np.random.choice(kp.shape[1], self.random_sample, replace=False)
                kp = kp[:, random_sample_kp, :]

            # affine transform
            if self.affine_aug:
                # prob of affine transform
                if self.affine_aug != None:
                    aff_prob = np.random.uniform(0, 1)
                else:
                    aff_prob = 0.0

                A = torch.randn(3, 4) * self.affine_param + torch.eye(3, 4)

                affine = F.affine_grid(A.unsqueeze(0), (1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), align_corners=True)

                # normalize kp
                if aff_prob > self.affine_prob:
                    (
                        img, 
                        mask,
                        multiple_mask,
                        kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(img).squeeze(0).float(), 
                        torch.from_numpy(mask).squeeze(0).float(), 
                        torch.from_numpy(multiple_mask).float(), 
                        torch.from_numpy(kp).squeeze(0).float(),
                        affine,
                        A
                    )
            
            # flip transform
            if self.flip_aug:
                # prob of flip transform
                prob = np.random.uniform(0, 1)
                if self.flip_aug != None:
                    flip_prob = prob
                else:
                    flip_prob = 0.0

                flip_axis = self.flip_axis

                # normalize kp
                if flip_prob > self.flip_prob:
                    (
                        img, 
                        mask,
                        multiple_mask,
                        kp
                    ) = self.__apply_flip_augmentation(
                        img[0], 
                        mask[0],
                        multiple_mask,
                        kp[0],
                        flip_axis
                    )
        return img, kp, mask, multiple_mask, labels
            
    def __standardize_orientation(self, img, mask, multiple_mask, kp, flip_axis):
        flip_kp = kp.copy() 
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            so_multiple_mask = np.zeros_like(multiple_mask)
            for i in range(multiple_mask.shape[0]):
                so_multiple_mask[i:i+1] = np.flip(multiple_mask[i], axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, so_multiple_mask, flip_kp
    
    def __apply_flip_augmentation(self, img, mask, multiple_mask, kp, flip_axis):
        flip_kp = kp.copy()
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_multiple_mask = np.zeros_like(multiple_mask)
            for i in range(multiple_mask.shape[0]):
                flip_multiple_mask[i:i+1] = np.flip(multiple_mask[i], axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_multiple_mask, flip_kp

    def __apply_affine_augmentation(self, img, mask, multiple_mask, kp, affine, A):
        # keypoint
        kp = (kp / torch.tensor([self.H // self.downsample, self.W // self.downsample, self.D // self.downsample]) * 2 - 1).flip(-1)

        kp = (torch.solve(torch.cat((kp, torch.ones(kp.shape[0], 1)), 1).float().t(),
              torch.cat((A, torch.tensor([0, 0, 0, 1]).view(1, -1)), 0))[0].t()[:, :3].squeeze())

        kp = ((kp + 1) / 2 * torch.tensor([self.H // self.downsample, self.W // self.downsample, self.D // self.downsample])).flip(-1)
        kp = kp.unsqueeze(0).numpy()

        # img
        img = F.grid_sample(img.view(1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        img = img.unsqueeze(0).numpy()

        # mask
        mask = F.grid_sample(mask.view(1, 1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        mask = mask.unsqueeze(0).numpy()
        
        # multiple mask
        aff_multiple_mask = F.grid_sample(multiple_mask.view(1, multiple_mask.shape[0], self.H // self.downsample, self.W // self.downsample, self.D // self.downsample), affine, align_corners=True).squeeze()
        aff_multiple_mask = aff_multiple_mask.unsqueeze(0).numpy()
        return img, mask, aff_multiple_mask, kp

    def __load_mask_dir(self, img_path, side=None): 
        mask_dir = img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        
        # extract and check the available masks in nifti
        all_nifti = natsorted([i.split(".")[0] for i in os.listdir(mask_dir) if "nii" in i])
        # filter out masks based on requirements (organ, side or specific region)
        if self.organs or side or self.specific_regions:
            selected_nifti = []
            if self.specific_regions:
                selected_nifti_sp_reg = [f for f in all_nifti if any([sr in f for sr in self.specific_regions])]
                selected_nifti += selected_nifti_sp_reg
            if self.organs:
                selected_nifti_organs = [f for f in all_nifti if any([organ in f for organ in self.organs])]
                if side:
                    selected_nifti_organs = [f for f in selected_nifti_organs if any([s in f for s in side])]
                selected_nifti += selected_nifti_organs
            all_nifti = selected_nifti
        
        # combine to be one mask for usage 
        mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        if self.organs and self.specific_regions:
            mask_class = len(self.organs) + len(self.specific_regions)
        elif self.organs:
            mask_class = len(self.organs)
        else:
            mask_class = len(all_nifti)
        multiple_mask = np.zeros((mask_class, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        for i, nif in enumerate(all_nifti):
            this_nif_mask = self.__load_nii_img(
                os.path.join(mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]

            mask += this_nif_mask
           
            if self.organs and self.specific_regions:
                for j in range(mask_class):
                    # specific region higher priority
                    if nif in self.specific_regions:
                        multiple_mask[j:j+1] += this_nif_mask
                    elif nif.split("_")[0] in self.organs:
                        multiple_mask[j:j+1] += this_nif_mask
                    else:
                        raise ValueError(f"{nif} not in selected list!")

            elif self.organs:
                for j, organ in enumerate(self.organs):
                    if organ in nif:
                        multiple_mask[j:j+1] += this_nif_mask
            else:
                multiple_mask[i:i+1] += this_nif_mask

        label_list = all_nifti if not self.organs else self.organs
        return mask, multiple_mask, label_list

    @staticmethod
    def __filter_mask_based_kpt(kpt, mask, label=1):
        kpt = kpt / 2
        H, W, D = mask.shape
        kpt_idx = np.round(kpt).astype(np.int16)
        kpt_idx[:, 0][kpt_idx[:, 0] >= H] = H-1
        kpt_idx[:, 1][kpt_idx[:, 1] >= W] = W-1
        kpt_idx[:, 2][kpt_idx[:, 2] >= D] = D-1
        kpt_mask = mask[kpt_idx[:, 0], kpt_idx[:, 1], kpt_idx[:, 2]]

        filter_idx = np.where(kpt_mask == label)[0]
        return kpt[filter_idx]*2

    @staticmethod
    def __load_nii_img(img_path, preprocess: bool = False, downsample: int = 1) -> np.ndarray:
        img = nib.load(img_path)
        arr = img.get_fdata()

        if preprocess:
            # clip pixel values
            min_bound = -1000.0
            max_bound = 500.0
            arr[arr < min_bound] = min_bound
            arr[arr > max_bound] = max_bound

            # normalize
            arr = (arr - min_bound) / (max_bound - min_bound)

        if downsample != 1:
            arr = scipy.ndimage.zoom(arr, zoom=(1/downsample, 1/downsample, 1/downsample), order=0)

        return arr
