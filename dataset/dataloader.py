import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
import os
import json
import numpy as np


class NLSTDataset(Dataset):
    def __init__(self, data_dir: str, json_file: str, mode: str = "train") -> None:
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode

        # read json file
        with open(os.path.join(data_dir, json_file)) as jf:
            file_contents = jf.read()
            jdict = json.loads(file_contents)

        # get subjects based on mode
        if mode == "train":
            subjects = jdict["training_paired_images"]
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

        fixed_img_path = os.path.join(self.data_dir, fixed_relative_path)
        moving_img_path = os.path.join(self.data_dir, moving_relative_path)

        fixed_img = self.__load_nii_img(fixed_img_path, preprocess=True)[None, ...]
        moving_img = self.__load_nii_img(moving_img_path, preprocess=True)[None, ...]

        # load fixed and moving keypoints
        fixed_kp = np.genfromtxt(
            fixed_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
            delimiter=",",
        )[None, ...]
        moving_kp = np.genfromtxt(
            moving_img_path.replace("images", "keypoints").replace("nii.gz", "csv"),
            delimiter=",",
        )[None, ...]

        # load masks
        fixed_mask = self.__load_nii_img(
            fixed_img_path.replace("images", "masks"), preprocess=False
        )[None, ...]
        moving_mask = self.__load_nii_img(
            moving_img_path.replace("images", "masks"), preprocess=False
        )[None, ...]

        return fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask

    @staticmethod
    def __load_nii_img(img_path, preprocess=False):
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

        return arr
