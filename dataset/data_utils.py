import torchio as tio
import numpy as np
import torch

def augmentations(augs, p=0.5):
    aug_process = []
    for aug in augs:
        if aug == "blur":
            aug_process.append(
                tio.RandomBlur(
                    p=p,
                    std=(0, 2),
                )
            )
        elif aug == "noise":
            aug_process.append(
                tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            )
            aug_process.append(
                tio.RandomNoise(
                    p=p,
                    mean=0,
                    std=(0, 0.25),
                )
            )
        elif aug == "transform":
            # note: the transformation augmentation here are not designed for registration task
            #       some keypoints will be missing if applying this!
            aug_process.append(
                tio.OneOf({
                    tio.RandomAffine(
                        scales=(0.001, 0.001, 0.001),
                        degrees=(5, 5, 5),
                        translation=(0.05, 0.05, 0.05),
                        image_interpolation='linear',
                        label_interpolation='nearest'
                    ): 1.0,
                    tio.RandomElasticDeformation(
                        num_control_points=(7, 7, 7),
                        max_displacement=7.5,
                        locked_borders=2,
                        image_interpolation='linear',
                        label_interpolation='nearest'
                    ): 0.0,
                }, p=p)
            )
        elif aug == "anisotropy":
            aug_process.append(
                tio.RandomAnisotropy(
                    p=p,
                    axes=(0, 1, 2),
                    downsampling=2,
                    image_interpolation='linear'
                )
            )
        elif aug == "flip":
            # note: the transformation augmentation here are not designed for registration task
            #       some keypoints will be misaligned if applying this!
            aug_process.append(
                tio.RandomFlip(
                    p=p,
                    axes=('LR',),
                    flip_probability=1.0,
                )
            )
        elif aug == "swap":
            # note: the transformation augmentation here are not designed for registration task
            #       some keypoints will be misaligned if applying this!
            aug_process.append(
                tio.RandomSwap(
                    p=p,
                    patch_size=15,
                    num_iterations=100,
                )
            )
        elif aug == "contrast":
            aug_process.append(
                tio.RandomGamma(
                    p=p,
                    log_gamma=(-0.3, 0.3),
                )
            )
    return aug_process



def torch2torchiodataset(Dataset, aug_process, transform="same", downsample=1, mae_pretrain=False):
    if mae_pretrain:
        subjects = []
        for i in range(len(Dataset)):
            img, kp, mask, multiple_mask, labels = Dataset[i]
            subject = tio.Subject(
                img=tio.ScalarImage(tensor=torch.from_numpy(img)),
                mask=tio.LabelMap(tensor=torch.from_numpy(mask)),
                mulmask=tio.LabelMap(tensor=torch.from_numpy(multiple_mask)),
                kpt=kp,
                labels=labels,
            )
            subjects.append(subject)
        subject_dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(aug_process), load_getitem=False)

    else:
        if transform == "same":
            subjects = []
            for i in range(len(Dataset)):
                fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, labels = Dataset[i]
                
                subject = tio.Subject(
                    f_img=tio.ScalarImage(tensor=torch.from_numpy(fixed_img)),
                    f_mask=tio.LabelMap(tensor=torch.from_numpy(fixed_mask)),
                    f_mulmask=tio.LabelMap(tensor=torch.from_numpy(fixed_multiple_mask)),
                    f_kpt=fixed_kp,
                    m_img=tio.ScalarImage(tensor=torch.from_numpy(moving_img)),
                    m_mask=tio.LabelMap(tensor=torch.from_numpy(moving_mask)),
                    m_mulmask=tio.LabelMap(tensor=torch.from_numpy(moving_multiple_mask)),
                    m_kpt=moving_kp,
                    labels=labels,
                )
                subjects.append(subject)
            subject_dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(aug_process), load_getitem=False)

        elif transform == "diff":
            fixed_subjects, moving_subjects = [], []
            for i in range(len(Dataset)):
                fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, labels = Dataset[i]
                fixed_subject = tio.Subject(
                    f_img=tio.ScalarImage(tensor=torch.from_numpy(fixed_img)),
                    f_mask=tio.LabelMap(tensor=torch.from_numpy(fixed_mask)),
                    f_mulmask=tio.LabelMap(tensor=torch.from_numpy(fixed_multiple_mask)),
                    f_kpt=fixed_kp,
                    labels=labels,
                )
                moving_subject = tio.Subject(
                    m_img=tio.ScalarImage(tensor=torch.from_numpy(moving_img)),
                    m_mask=tio.LabelMap(tensor=torch.from_numpy(moving_mask)),
                    m_mulmask=tio.LabelMap(tensor=torch.from_numpy(moving_multiple_mask)),
                    m_kpt_img=moving_kp,
                    labels=labels,
                )
                fixed_subjects.append(fixed_subject)
                moving_subjects.append(moving_subject)
                
            subject_dataset = (
                tio.SubjectsDataset(fixed_subjects, transform=tio.Compose(aug_process), load_getitem=False),
                tio.SubjectsDataset(moving_subjects, transform=tio.Compose(aug_process), load_getitem=False),
            )

    return subject_dataset

