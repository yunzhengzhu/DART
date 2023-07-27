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
            aug_process.append(
                tio.RandomFlip(
                    p=p,
                    axes=(0, 1, 2),
                    flip_probability=0.5,
                )
            )
        elif aug == "swap":
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



def torch2torchiodataset(Dataset, aug_process, transform="same", downsample=1):
    if transform == "same":
        subjects = []
        for i in range(len(Dataset)):
            fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask = Dataset[i]
            #print(f"Original Keypoints: fixed {fixed_kp.shape} moving {moving_kp.shape}")
            fixed_kp_img = np.zeros_like(fixed_img)
            moving_kp_img = np.zeros_like(moving_img)
            #fixed_kp_img = np.zeros((fixed_img.shape[0], Dataset.H, Dataset.W, Dataset.D))
            #moving_kp_img = np.zeros((moving_img.shape[0], Dataset.H, Dataset.W, Dataset.D))
            if downsample > 1:
                fixed_kp = fixed_kp // downsample
                moving_kp = moving_kp // downsample

            for kp_id, (f_kp, m_kp) in enumerate(zip(fixed_kp.squeeze().astype(np.int64), moving_kp.squeeze().astype(np.int64))):
                fixed_kp_img[:, f_kp[0], f_kp[1], f_kp[2]] = kp_id + 1
                moving_kp_img[:, m_kp[0], m_kp[1], m_kp[2]] = kp_id + 1

            subject = tio.Subject(
                f_img=tio.ScalarImage(tensor=torch.from_numpy(fixed_img)),
                f_mask=tio.LabelMap(tensor=torch.from_numpy(fixed_mask)),
                f_kpt_img=tio.LabelMap(tensor=torch.from_numpy(fixed_kp_img)),
                m_img=tio.ScalarImage(tensor=torch.from_numpy(moving_img)),
                m_mask=tio.LabelMap(tensor=torch.from_numpy(moving_mask)),
                m_kpt_img=tio.LabelMap(tensor=torch.from_numpy(moving_kp_img))
            )
            subjects.append(subject)
        subject_dataset = tio.SubjectsDataset(subjects, transform=tio.Compose(aug_process), load_getitem=False)

    elif transform == "diff":
        fixed_subjects, moving_subjects = [], []
        for i in range(len(Dataset)):
            fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask = Dataset[i]
            #print(f"Original Keypoints: fixed {fixed_kp.shape} moving {moving_kp.shape}")
            fixed_kp_img = np.zeros_like(fixed_img)
            moving_kp_img = np.zeros_like(moving_img)
            if downsample > 1:
                fixed_kp = fixed_kp // downsample
                moving_kp = moving_kp // downsample
            for kp_id, (f_kp, m_kp) in enumerate(zip(fixed_kp.squeeze().astype(np.int64), moving_kp.squeeze().astype(np.int64))):
                fixed_kp_img[:, f_kp[0], f_kp[1], f_kp[2]] = kp_id + 1
                moving_kp_img[:, m_kp[0], m_kp[1], m_kp[2]] = kp_id + 1
            fixed_subject = tio.Subject(
                f_img=tio.ScalarImage(tensor=torch.from_numpy(fixed_img)),
                f_mask=tio.LabelMap(tensor=torch.from_numpy(fixed_mask)),
                f_kpt_img=tio.LabelMap(tensor=torch.from_numpy(fixed_kp_img))
            )
            moving_subject = tio.Subject(
                m_img=tio.ScalarImage(tensor=torch.from_numpy(moving_img)),
                m_mask=tio.LabelMap(tensor=torch.from_numpy(moving_mask)),
                m_kpt_img=tio.LabelMap(tensor=torch.from_numpy(moving_kp_img))
            )
            fixed_subjects.append(fixed_subject)
            moving_subjects.append(moving_subject)
            
        subject_dataset = (
            tio.SubjectsDataset(fixed_subjects, transform=tio.Compose(aug_process), load_getitem=False),
            tio.SubjectsDataset(moving_subjects, transform=tio.Compose(aug_process), load_getitem=False),
        )

    return subject_dataset

