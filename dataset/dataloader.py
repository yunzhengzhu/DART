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
from model.transform import AffineTransform
from affine import Affine
import random


class NLSTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        json_file: str,
        mode: str = "train",
        downsample: int = 1,
        preprocess: bool = False,
        random_sample: int = None,
        affine_aug: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode
        self.downsample = downsample
        self.preprocess = preprocess
        self.random_sample = random_sample
        self.affine_aug = affine_aug

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
            subjects = [
                subject
                for subject in subjects
                if subject not in jdict["registration_val"]
            ]
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

        fixed_img = self.__load_nii_img(
            fixed_img_path, preprocess=self.preprocess, downsample=self.downsample
        )  # [None, ...]
        moving_img = self.__load_nii_img(
            moving_img_path, preprocess=self.preprocess, downsample=self.downsample
        )  # [None, ...]

        # load fixed and moving keypoints

        fixed_kp = np.genfromtxt(
            fixed_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
            delimiter=",",
        )  # [None, ...] #/ self.downsample

        moving_kp = np.genfromtxt(
            moving_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
            delimiter=",",
        )  # [None, ...] #/ self.downsample

        # load masks
        fixed_mask = self.__load_nii_img(
            fixed_img_path.replace("images", "masks"),
            preprocess=False,
            downsample=self.downsample,
        )  # [None, ...]
        moving_mask = self.__load_nii_img(
            moving_img_path.replace("images", "masks"),
            preprocess=False,
            downsample=self.downsample,
        )  # [None, ...]

        # transform from numpy to torch tensor
        fixed_img = torch.tensor(fixed_img).float()
        moving_img = torch.tensor(moving_img).float()
        fixed_kp = torch.tensor(fixed_kp).float()
        moving_kp = torch.tensor(moving_kp).float()
        fixed_mask = torch.tensor(fixed_mask).float()
        moving_mask = torch.tensor(moving_mask).float()

        if self.mode == "train":
            # randomly sample keypoints
            if self.random_sample:
                random_sample_kp = np.random.choice(
                    fixed_kp.shape[0], self.random_sample, replace=False
                )
                fixed_kp = fixed_kp[random_sample_kp, :]
                moving_kp = moving_kp[random_sample_kp, :]

            # affine transform
            if self.affine_aug:
                # A = torch.randn(3, 4) * 0.035 + torch.eye(3, 4)
                #only apply translation
                A = torch.cat(
                    (
                        torch.eye(3, 3),
                        torch.tensor([[random.uniform(-0.1, 0.1), 0, 0]]).t(),
                    ),
                    1,
                )
                affine = F.affine_grid(
                    A.unsqueeze(0),
                    (
                        1,
                        1,
                        self.H // self.downsample,
                        self.W // self.downsample,
                        self.D // self.downsample,
                    ),
                    align_corners=True,
                )
                # normalize kp
                norm_moving_kp = (
                    moving_kp
                    / torch.tensor(
                        [
                            self.H // self.downsample,
                            self.W // self.downsample,
                            self.D // self.downsample,
                        ]
                    )
                    * 2
                    - 1
                ).flip(-1)

                # apply affine
                moving_kp = (
                    torch.solve(
                        torch.cat(
                            (norm_moving_kp, torch.ones(norm_moving_kp.shape[0], 1)), 1
                        )
                        .float()
                        .t(),
                        torch.cat((A, torch.tensor([0, 0, 0, 1]).view(1, -1)), 0),
                    )[0]
                    .t()[:, :3]
                    .squeeze()
                )

                # denormalize kp
                moving_kp = (
                    (moving_kp + 1)
                    / 2
                    * torch.tensor(
                        [
                            self.H // self.downsample,
                            self.W // self.downsample,
                            self.D // self.downsample,
                        ]
                    )
                ).flip(-1)

                #apply affine to image and mask
                moving_img = F.grid_sample(
                    moving_img.view(
                        1,
                        1,
                        self.H // self.downsample,
                        self.W // self.downsample,
                        self.D // self.downsample,
                    ),
                    affine,
                    align_corners=True,
                ).squeeze()

                moving_mask = F.grid_sample(
                    moving_mask.view(
                        1,
                        1,
                        self.H // self.downsample,
                        self.W // self.downsample,
                        self.D // self.downsample,
                    ),
                    affine,
                    align_corners=True,
                ).squeeze()

        fixed_img = fixed_img.unsqueeze(0)
        moving_img = moving_img.unsqueeze(0)
        fixed_mask = fixed_mask.unsqueeze(0)
        moving_mask = moving_mask.unsqueeze(0)
        fixed_kp = fixed_kp.unsqueeze(0)
        moving_kp = moving_kp.unsqueeze(0)

        return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask

    @staticmethod
    def __load_nii_img(
        img_path, preprocess: bool = False, downsample: int = 1
    ) -> np.ndarray:
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
            arr = scipy.ndimage.zoom(
                arr, zoom=(1 / downsample, 1 / downsample, 1 / downsample), order=0
            )

        return arr
