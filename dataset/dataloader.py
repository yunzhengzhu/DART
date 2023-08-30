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

        # get subjects based on mode
        if mode == "train":
            subjects = jdict["training_paired_images"]
            # remove duplicate cases from val
            subjects = [subject for subject in subjects if subject not in jdict["registration_val"]]
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
        
        if self.texture_mask_dir:
            # texture mask
            fixed_texture_mask = self.__load_nii_img(
                fixed_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
            moving_texture_mask = self.__load_nii_img(
                moving_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
        
        # load masks
        if (self.mode == "train" and self.mask_dir) or (self.mode == "val" and self.eval_with_mask):
            if self.side == ["both"]:
                side = self.subjects[idx]["side"]
            else:
                side = self.side
            #print(f"original: {fixed_kp.shape} {moving_kp.shape}")
            fixed_mask, moving_mask = self.__load_mask_dir(fixed_img_path, moving_img_path, side=side)
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
                    fixed_kp,
                ) = self.__standardize_orientation(
                    fixed_img[0],
                    fixed_mask[0],
                    fixed_kp[0],
                    flip_axis=flip_axis,
                )
                
                (
                    moving_img,
                    moving_mask,
                    moving_kp,
                ) = self.__standardize_orientation(
                    moving_img[0],
                    moving_mask[0],
                    moving_kp[0],
                    flip_axis=flip_axis,
                )

        if self.mode == 'train':
            # randomly sample keypoints
            if self.random_sample != None:
                random_sample_kp = np.random.choice(fixed_kp.shape[1], self.random_sample, replace=False)
                fixed_kp = fixed_kp[:, random_sample_kp, :]
                moving_kp = moving_kp[:, random_sample_kp, :]

            # affine transform
            if self.affine_aug:
                # prob of affine transform
                if self.affine_aug == "moving" or "both":
                    moving_aff_prob = np.random.uniform(0, 1)
                else:
                    moving_aff_prob = 0.0

                if self.affine_aug == "fixed" or "both":
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
                        fixed_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(fixed_img).squeeze(0).float(), 
                        torch.from_numpy(fixed_mask).squeeze(0).float(), 
                        torch.from_numpy(fixed_kp).squeeze(0).float(),
                        affine,
                        A
                    )
                if moving_aff_prob > self.affine_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(moving_img).squeeze(0).float(), 
                        torch.from_numpy(moving_mask).squeeze(0).float(), 
                        torch.from_numpy(moving_kp).squeeze(0).float(), 
                        affine,
                        A
                    )
            
            # flip transform
            if self.flip_aug:
                # prob of flip transform
                prob = np.random.uniform(0, 1)
                if self.flip_aug == "moving" or "both":
                    moving_flip_prob = prob
                else:
                    moving_flip_prob = 0.0

                if self.flip_aug == "fixed" or "both":
                    fixed_flip_prob = prob
                else:
                    fixed_flip_prob = 0.0

                flip_axis = self.flip_axis

                # normalize kp
                if fixed_flip_prob > self.flip_prob:
                    (
                        fixed_img, 
                        fixed_mask, 
                        fixed_kp
                    ) = self.__apply_flip_augmentation(
                        fixed_img[0], 
                        fixed_mask[0], 
                        fixed_kp[0],
                        flip_axis
                    )
                if moving_flip_prob > self.flip_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_kp
                    ) = self.__apply_flip_augmentation(
                        moving_img[0], 
                        moving_mask[0], 
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
        
        return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask
            
    def __standardize_orientation(self, img, mask, kp, flip_axis):
        flip_kp = kp.copy() 
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_kp
    
    def __apply_flip_augmentation(self, img, mask, kp, flip_axis):
        flip_kp = kp.copy()
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_kp

    def __apply_affine_augmentation(self, img, mask, kp, affine, A):
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

        return img, mask, kp
       

    def __load_mask_dir(self, fixed_img_path, moving_img_path, side=None): 
        fixed_mask_dir = fixed_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        moving_mask_dir = moving_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        
        # extract and check the available masks in nifti
        all_nifti_f = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])
        all_nifti_m = natsorted([i.split(".")[0] for i in os.listdir(moving_mask_dir) if "nii" in i])
        assert all_nifti_f == all_nifti_m
        all_nifti = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])

        # filter out masks based on requirements (organ, side, specific lobe)
        if self.organs or self.side or self.specific_regions:
            selected_nifti = [f for f in all_nifti if any([organ in f for organ in self.organs])]
            if side:
                selected_nifti = [f for f in selected_nifti if any([s in f for s in side])]
            if self.specific_regions:
                selected_nifti = [f for f in selected_nifti if any([sr in f for sr in self.specific_regions])]
           
            all_nifti = selected_nifti

        # combine to be one mask for usage 
        fixed_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        moving_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        for nif in all_nifti:
            this_nif_fixed_mask = self.__load_nii_img(
                os.path.join(fixed_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            fixed_mask += this_nif_fixed_mask
            
            this_nif_moving_mask = self.__load_nii_img(
                os.path.join(moving_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            moving_mask += this_nif_moving_mask
        
        return fixed_mask, moving_mask
    
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


class NLSTDataset_MAEPretrain(Dataset):
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

        # get subjects based on mode
        if mode == "train":
            subjects = jdict["training_paired_images"]
            # remove duplicate cases from val
            subjects = [subject for subject in subjects if subject not in jdict["registration_val"]]
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
        import pdb
        pdb.set_trace()
        
    def __len__(self) -> int:
        return len(self.subjects) * 2

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
        
        if self.texture_mask_dir:
            # texture mask
            fixed_texture_mask = self.__load_nii_img(
                fixed_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
            moving_texture_mask = self.__load_nii_img(
                moving_img_path.replace("images", "masks").replace("masksTr", self.texture_mask_dir), preprocess=False, downsample=self.downsample
            )[None, ...]
        
        # load masks
        if (self.mode == "train" and self.mask_dir) or (self.mode == "val" and self.eval_with_mask):
            if self.side == ["both"]:
                side = self.subjects[idx]["side"]
            else:
                side = self.side
            #print(f"original: {fixed_kp.shape} {moving_kp.shape}")
            fixed_mask, moving_mask = self.__load_mask_dir(fixed_img_path, moving_img_path, side=side)
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
                    fixed_kp,
                ) = self.__standardize_orientation(
                    fixed_img[0],
                    fixed_mask[0],
                    fixed_kp[0],
                    flip_axis=flip_axis,
                )
                
                (
                    moving_img,
                    moving_mask,
                    moving_kp,
                ) = self.__standardize_orientation(
                    moving_img[0],
                    moving_mask[0],
                    moving_kp[0],
                    flip_axis=flip_axis,
                )

        if self.mode == 'train':
            # randomly sample keypoints
            if self.random_sample != None:
                random_sample_kp = np.random.choice(fixed_kp.shape[1], self.random_sample, replace=False)
                fixed_kp = fixed_kp[:, random_sample_kp, :]
                moving_kp = moving_kp[:, random_sample_kp, :]

            # affine transform
            if self.affine_aug:
                # prob of affine transform
                if self.affine_aug == "moving" or "both":
                    moving_aff_prob = np.random.uniform(0, 1)
                else:
                    moving_aff_prob = 0.0

                if self.affine_aug == "fixed" or "both":
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
                        fixed_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(fixed_img).squeeze(0).float(), 
                        torch.from_numpy(fixed_mask).squeeze(0).float(), 
                        torch.from_numpy(fixed_kp).squeeze(0).float(),
                        affine,
                        A
                    )
                if moving_aff_prob > self.affine_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_kp
                    ) = self.__apply_affine_augmentation(
                        torch.from_numpy(moving_img).squeeze(0).float(), 
                        torch.from_numpy(moving_mask).squeeze(0).float(), 
                        torch.from_numpy(moving_kp).squeeze(0).float(), 
                        affine,
                        A
                    )
            
            # flip transform
            if self.flip_aug:
                # prob of flip transform
                prob = np.random.uniform(0, 1)
                if self.flip_aug == "moving" or "both":
                    moving_flip_prob = prob
                else:
                    moving_flip_prob = 0.0

                if self.flip_aug == "fixed" or "both":
                    fixed_flip_prob = prob
                else:
                    fixed_flip_prob = 0.0

                flip_axis = self.flip_axis

                # normalize kp
                if fixed_flip_prob > self.flip_prob:
                    (
                        fixed_img, 
                        fixed_mask, 
                        fixed_kp
                    ) = self.__apply_flip_augmentation(
                        fixed_img[0], 
                        fixed_mask[0], 
                        fixed_kp[0],
                        flip_axis
                    )
                if moving_flip_prob > self.flip_prob:
                    (
                        moving_img, 
                        moving_mask, 
                        moving_kp
                    ) = self.__apply_flip_augmentation(
                        moving_img[0], 
                        moving_mask[0], 
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
        
        return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask
            
    def __standardize_orientation(self, img, mask, kp, flip_axis):
        flip_kp = kp.copy() 
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_kp
    
    def __apply_flip_augmentation(self, img, mask, kp, flip_axis):
        flip_kp = kp.copy()
        for fa in flip_axis:
            img = np.flip(img, axis=fa).copy()
            mask = np.flip(mask, axis=fa).copy()
            flip_kp[:, fa] = (img.shape[fa] * self.downsample - 1) - flip_kp[:, fa]
        img = img[None, ...]
        mask = mask[None, ...]
        flip_kp = flip_kp[None, ...]
        return img, mask, flip_kp

    def __apply_affine_augmentation(self, img, mask, kp, affine, A):
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

        return img, mask, kp
       

    def __load_mask_dir(self, fixed_img_path, moving_img_path, side=None): 
        fixed_mask_dir = fixed_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        moving_mask_dir = moving_img_path.replace("images", "masks").split(os.extsep)[0].replace("masksTr", self.mask_dir)
        
        # extract and check the available masks in nifti
        all_nifti_f = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])
        all_nifti_m = natsorted([i.split(".")[0] for i in os.listdir(moving_mask_dir) if "nii" in i])
        assert all_nifti_f == all_nifti_m
        all_nifti = natsorted([i.split(".")[0] for i in os.listdir(fixed_mask_dir) if "nii" in i])
        
        # filter out masks based on requirements (organ, side or specific region)
        if self.organs or self.side or self.specific_regions:
            if self.specific_regions:
                selected_nifti = [f for f in selected_nifti if any([sr in f for sr in self.specific_regions])]
            else:
                selected_nifti = [f for f in all_nifti if any([organ in f for organ in self.organs])]
                if side:
                    selected_nifti = [f for f in selected_nifti if any([s in f for s in side])]
           
            all_nifti = selected_nifti
        
        # combine to be one mask for usage 
        fixed_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        moving_mask = np.zeros((1, self.H // self.downsample, self.W // self.downsample, self.D // self.downsample))
        for nif in all_nifti:
            this_nif_fixed_mask = self.__load_nii_img(
                os.path.join(fixed_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            fixed_mask += this_nif_fixed_mask
            
            this_nif_moving_mask = self.__load_nii_img(
                os.path.join(moving_mask_dir, nif+".nii.gz"), preprocess=False, downsample=self.downsample
            )[None, ...]
            
            moving_mask += this_nif_moving_mask
        
        return fixed_mask, moving_mask
    
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
