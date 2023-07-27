import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import os
import json
import numpy as np
import scipy
from typing import Tuple


class NLSTDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        json_file: str, 
        mode: str = "train", 
        downsample: int = 1, 
        preprocess: bool = False, 
        random_sample: int = None, 
        kp_dir: str = None, 
        affine_aug: str = None,
        affine_prob: float = 0.5,
        affine_param: float = 0.035
    ) -> None:
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode
        self.downsample = downsample
        self.preprocess = preprocess
        self.random_sample = random_sample
        self.kp_dir = kp_dir
        self.affine_aug = affine_aug
        self.affine_prob = affine_prob
        self.affine_param = affine_param

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
        
        # load masks
        fixed_mask = self.__load_nii_img(
            fixed_img_path.replace("images", "masks"), preprocess=False, downsample=self.downsample
        )[None, ...]
        moving_mask = self.__load_nii_img(
            moving_img_path.replace("images", "masks"), preprocess=False, downsample=self.downsample
        )[None, ...]
        
        # mask
        #fixed_img[~fixed_mask.astype(bool)] = fixed_img.min()
        #moving_img[~moving_mask.astype(bool)] = moving_img.min()

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

        return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask

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

