import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, mov_image, flow, mod="bilinear"):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [
                torch.linspace(-1, 1, d2),
                torch.linspace(-1, 1, h2),
                torch.linspace(-1, 1, w2),
            ]
        )
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:, :, :, :, 0]
        flow_h = flow[:, :, :, :, 1]
        flow_w = flow[:, :, :, :, 2]
        
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = F.grid_sample(
            mov_image, sample_grid, mode=mod, align_corners=True
        )
        
        return warped


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid(
            [
                torch.linspace(-1, 1, d2),
                torch.linspace(-1, 1, h2),
                torch.linspace(-1, 1, w2),
            ]
        )
        grid_h = grid_h.cuda().float()
        grid_d = grid_d.cuda().float()
        grid_w = grid_w.cuda().float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2**self.time_step)
        
        for i in range(self.time_step):
            flow_d = flow[:, 0, :, :, :]
            flow_h = flow[:, 1, :, :, :]
            flow_w = flow[:, 2, :, :, :]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)

            deformation = torch.stack(
                (disp_w, disp_h, disp_d), 4
            )  # shape (N, D, H, W, 3)
            
            flow = flow + F.grid_sample(
                flow,
                deformation,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        return flow

class ResizeTransform(nn.Module):
    def __init__(self, factor=1):
        super(ResizeTransform, self).__init__()
        self.factor = factor

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode="trilinear")
        
        return x
