import torch
import torch.nn.functional as F
import numpy as np
import math
from utils.feature_utils import mindssc
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
        return torch.nn.MSELoss()(y_pred, y_true)

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
    def __call__(self, fix_lms, mov_lms, disp, spacing_fix, spacing_mov, downsample=1, normalize=True):
        mov_lms = mov_lms.squeeze().flip(-1) # row, 3
        fix_lms = fix_lms.squeeze().flip(-1) # row, 3
        H, W, D = disp.shape[2:]
        if normalize:
            mov_lms = mov_lms / torch.tensor([H*downsample, W*downsample, D*downsample]).cuda() * 2 - 1
            fix_lms = fix_lms / torch.tensor([H*downsample, W*downsample, D*downsample]).cuda() * 2 - 1
        gt_lmdiff = mov_lms - fix_lms

        pred_lmsdiff = F.grid_sample(disp, fix_lms.view(1, -1, 1, 1, 3), align_corners=True, mode='bilinear').squeeze().t()
        return torch.nn.MSELoss()(pred_lmsdiff, gt_lmdiff)

class Seg_MSE:
    """
    Mean sqaured error loss for segmentation.
    """
   
    def __call__(self, y_pred, y_true):
        return torch.nn.MSELoss()(y_pred, y_true)
        

class kpt_Sim:
    """
    Similarity on the kpt centered window
    """
    def __init__(self, win=9, metric="ncc", eps=1e-5):
        self.win = win
        self.metric = metric
        self.eps = eps

    def __call__(self, fix_img, mov_img, fix_kpts, mov_kpts, downsample=1):
        device = fix_img.device
        if downsample > 1:
            fix_img = F.interpolate(fix_img, scale_factor=downsample, mode='trilinear', align_corners=False)
            mov_img = F.interpolate(mov_img, scale_factor=downsample, mode='trilinear', align_corners=False)
        fix_img = fix_img[0, ...].to(device)
        mov_img = mov_img[0, ...].to(device)
        fix_kpts = fix_kpts[0, 0].to(device)
        mov_kpts = mov_kpts[0, 0].to(device)

        N, n_dim = fix_kpts.shape
        C, H, W, D = fix_img.shape

        fix_kpts = torch.round(fix_kpts).to(torch.long).to(device)
        mov_kpts = torch.round(mov_kpts).to(torch.long).to(device)

        win_size = torch.as_tensor(self.win**3).to(device)
        radius = self.win // 2
        [h, w, d] = torch.meshgrid(
            torch.linspace(-radius, radius, self.win), 
            torch.linspace(-radius, radius, self.win),
            torch.linspace(-radius, radius, self.win)
        )
        h = h.flatten().to(torch.long)
        w = w.flatten().to(torch.long)
        d = d.flatten().to(torch.long)

        fix_img_pad = F.pad(fix_img, pad=6*(radius,), value=0.).to(device)
        mov_img_pad = F.pad(mov_img, pad=6*(radius,), value=0.).to(device)

        score = torch.zeros(1).to(device)

        if self.metric == "ncc":
            I_sum = torch.zeros(C, N).to(device)
            I2_sum = torch.zeros(C, N).to(device)
            IJ_sum = torch.zeros(C, N).to(device)
            J_sum = torch.zeros(C, N).to(device)
            J2_sum = torch.zeros(C, N).to(device)
            uI = torch.zeros(C, 1)
            uJ = torch.zeros(C, 1)
            for (hh, ww, dd) in zip(h, w, d):
                hhh = hh + radius
                www = ww + radius
                ddd = dd + radius
                I_tmp = fix_img_pad[:, fix_kpts[:, 0] + hhh, fix_kpts[:, 1] + www, fix_kpts[:, 2] + ddd].reshape(C, -1).to(device)
                J_tmp = mov_img_pad[:, mov_kpts[:, 0] + hhh, mov_kpts[:, 1] + www, mov_kpts[:, 2] + ddd].reshape(C, -1).to(device)

                I_sum += I_tmp
                J_sum += J_tmp
                IJ_sum += I_tmp * J_tmp
                I2_sum += I_tmp ** 2
                J2_sum += J_tmp ** 2
            uI = I_sum / win_size
            uJ = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + self.eps)
            score = -1.0 * torch.mean(cc)
        
        elif self.metric == "mse":
            sub_channel_score = torch.zeros(C, N).to(device)
            for (hh, ww, dd) in zip(h, w, d):
                hhh = hh + radius
                www = ww + radius
                ddd = dd + radius
                I_tmp = fix_img_pad[:, fix_kpts[:, 0] + hhh, fix_kpts[:, 1] + www, fix_kpts[:, 2] + ddd].reshape(C, -1).to(device)
                J_tmp = mov_img_pad[:, mov_kpts[:, 0] + hhh, mov_kpts[:, 1] + www, mov_kpts[:, 2] + ddd].reshape(C, -1).to(device)
                sub_channel_score += (I_tmp - J_tmp) ** 2
            score = torch.mean(torch.mean(sub_channel_score, dim = 1))

        return score

class kpt_Sim_orig:
    """
    Similarity on the kpt centered window
    """
    def __init__(self, win=9, metric="ncc", eps=1e-5):
        self.win = win
        self.metric = metric
        self.eps = eps

    def __call__(self, fix_img, mov_img, fix_kpts, mov_kpts, downsample=1):
        kpt_sim_loss = []
        num_channels = fix_img.shape[1]
        for kpf, kpm in zip(fix_kpts.squeeze(), mov_kpts.squeeze()):
            this_channel_kpt_sim_loss = []
            for channel in range(num_channels):
                fix_img_patch = self.extract_keypoint_patch(
                    fix_img[:, channel:channel+1, ...],
                    kpf, 
                    self.win,
                    downsample=downsample
                )
                mov_img_patch = self.extract_keypoint_patch(
                    mov_img[:, channel:channel+1, ...],
                    kpm,
                    self.win,
                    downsample=downsample
                )
                score = self.compute_sim_score(
                    fix_img_patch.to(fix_img.device).to(torch.float32),
                    mov_img_patch.to(mov_img.device).to(torch.float32),
                    metric=self.metric
                )
                this_channel_kpt_sim_loss.append(score)
            kpt_sim_loss.append(sum(this_channel_kpt_sim_loss) / num_channels)
        return sum(kpt_sim_loss) / len(kpt_sim_loss)
        
    def compute_sim_score(self, I, J, metric="ncc"):
        if metric == "ncc":
            win_dims = [self.win] * 3
            weight = torch.ones(
                (1, 1, self.win, self.win, self.win),
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
            I_sum = torch.sum(I) #conv_fn(I, weight, padding=int(self.win / 2))
            J_sum = torch.sum(J) #conv_fn(J, weight, padding=int(self.win / 2))
            I2_sum = torch.sum(I2) #conv_fn(I2, weight, padding=int(self.win / 2))
            J2_sum = torch.sum(J2) #conv_fn(J2, weight, padding=int(self.win / 2))
            IJ_sum = torch.sum(IJ) #conv_fn(IJ, weight, padding=int(self.win / 2))

            # compute cross correlation
            win_size = torch.prod(torch.Tensor([I.shape[-3:]])) #np.prod(win_dims)

            u_I = I_sum / win_size
            u_J = J_sum / win_size

            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            cc = cross * cross / (I_var * J_var + self.eps)
            score = -1.0 * torch.mean(cc)

        elif metric == "mse":
            score = torch.mean((I - J) ** 2)

        return score


    def extract_keypoint_patch(self, img, kpt, win, downsample=1):
        H, W, D = img.shape[2:]
        ix, iy, iz = kpt
        ix = round(int(ix))
        iy = round(int(iy))
        iz = round(int(iz))
        if downsample > 1:
            H = H * downsample
            W = W * downsample
            D = D * downsample
            img = F.interpolate(img, scale_factor=downsample, mode='trilinear', align_corners=False)

        # x dim
        x_min = ix - win//2
        x_max = ix + win//2
        shiftx = 0
        if ix < win//2:
            shiftx = win//2 - ix # pos shift
        elif ix > (H - 1) - win//2:
            shiftx = (H - 1) - win//2 - ix # neg shift

        # y dim
        y_min = iy - win//2
        y_max = iy + win//2
        shifty = 0
        if iy < win//2:
            shifty = win//2 - iy
        elif iy > (W - 1) - win//2:
            shifty = (W - 1) - win//2 - iy

        # z dim
        z_min = iz - win//2
        z_max = iz + win//2
        shiftz = 0
        if iz < win//2:
            shiftz = win//2 - iz
        elif iz > (D - 1) - win//2:
            shiftz = (D - 1) - win//2 - iz
        
        patch = img[:, :, x_min+shiftx:x_max+shiftx+1, y_min+shifty:y_max+shifty+1, z_min+shiftz:z_max+shiftz+1]
        return patch

class MINDSSC:
    """
    MIND loss.
    """
    def __init__(self, loss_type="mse"):
        self.loss_type = loss_type

    def __call__(self, y_true, y_pred):
        mind_y_true = mindssc(y_true)
        mind_y_pred = mindssc(y_pred)
        if self.loss_type == "mse":
            return torch.mean((mind_y_true - mind_y_pred) ** 2)

        elif self.loss_type == "ncc":
            ncc = np.zeros(mind_y_true.shape[1])
            for channel in range(mind_y_true.shape[1]):
                ncc[channel] = NCC()(mind_y_true[:, channel:channel+1, ...], mind_y_pred[:, channel:channel+1, ...])
            
            return ncc.mean()
