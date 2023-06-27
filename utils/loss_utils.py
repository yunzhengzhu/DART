import torch
import torch.nn.functional as F
import numpy as np
import math
from scipy.ndimage import map_coordinates

"""
Adopted from https://github.com/xi-jia/LKU-Net/blob/main/train.py
"""


class smoothLoss:
    def __call__(self, y_pred):
        d2, h2, w2 = y_pred.shape[-3:]
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) / 2 * d2
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) / 2 * h2
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) / 2 * w2
        return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


"""
Normalized local cross-correlation (or Local NCC) function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones(
            (1, 1, weight_win_size, weight_win_size, weight_win_size),
            device=I.device,
            requires_grad=False,
        )
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


"""
Global normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""


class GNCC(torch.nn.Module):
    """
    global normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-5):
        super(GNCC, self).__init__()
        self.eps = eps

    def forward(self, I, J):
        ndims = 3

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute global sums
        I_sum = torch.sum(I)
        J_sum = torch.sum(J)
        I2_sum = torch.sum(I2)
        J2_sum = torch.sum(J2)
        IJ_sum = torch.sum(IJ)

        # compute cross correlation
        win_size = torch.prod(torch.Tensor([I.shape[-ndims:]]))
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * cc


class Dice:
    """
    N-D dice for segmentation
    """

    def __call__(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class MSE:
    """
    Mean squared error loss.
    """

    def __call__(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class SAD:
    """
    Absolute error loss.
    """

    def __call__(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class TRE:
    """
    Target registration error for keypoints
    """
    ## numpy version
    def __call__(self, fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
        tre = []
        for subject_id in range(len(disp)):
            fix_lms_disp_x = map_coordinates(disp[subject_id][:, :, :, 0], fix_lms[subject_id].transpose())
            fix_lms_disp_y = map_coordinates(disp[subject_id][:, :, :, 1], fix_lms[subject_id].transpose())
            fix_lms_disp_z = map_coordinates(disp[subject_id][:, :, :, 2], fix_lms[subject_id].transpose())
            fix_lms_disp = np.array(
                (fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)
            ).transpose()

            fix_lms_warped = fix_lms[subject_id] + fix_lms_disp
            tre.append(np.linalg.norm((fix_lms_warped - mov_lms[subject_id]) * spacing_mov, axis=2).mean(1))
        return np.mean(np.array(tre))
    
        
