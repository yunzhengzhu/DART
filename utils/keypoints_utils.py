import numpy as np
import torch
import torch.nn.functional as F
import scipy
import nibabel as nib
from matplotlib import pyplot as plt
from matplotlib import patches
def extract_range(kpt, win, shape=(224, 192, 224)):
    ix, iy, iz = kpt
    ix = round(ix)
    iy = round(iy)
    iz = round(iz)
    # x dim
    x_min = ix - win//2
    x_max = ix + win//2
    shiftx = 0
    if ix < win//2:
        #x_min = 0
        shiftx = win//2 - ix # pos shift
    elif ix > (shape[0] - 1) - win//2:
        #x_max = shape[0]
        shiftx = (shape[0] - 1) - win//2 - ix # neg shift
    
    # y dim
    y_min = iy - win//2
    y_max = iy + win//2
    shifty = 0
    if iy < win//2:
        #y_min = 0
        shifty = win//2 - iy
    elif iy > (shape[1] - 1) - win//2:
        #y_max = shape[1]
        shifty = (shape[1] - 1) - win//2 - iy
    
    # z dim
    z_min = iz - win//2
    z_max = iz + win//2
    shiftz = 0
    if iz < win//2:
        #z_min = 0
        shiftz = win//2 - iz
    elif iz > (shape[2] - 1) - win//2:
        #z_max = shape[2]
        shiftz = (shape[2] - 1) - win//2 - iz
        
    return x_min, x_max, y_min, y_max, z_min, z_max, shiftx, shifty, shiftz



def keypoint_score(img_fixed, img_moving, kpt_fixed, kpt_moving, score_metric='tre', win_len=9, eps=1e-5, device='cuda', verbose=False):
    cc_score, mse_score, tre_score = [], [], []
    for i, (kpt_f, kpt_m) in enumerate(zip(kpt_fixed, kpt_moving)):
        #print(f'Keypoint Fixed {kpt_f} Moving {kpt_m}')
#     i = 1842
#     kpt_f, kpt_m = kpt_fixed[i], kpt_moving[i]
        if score_metric != 'tre':
            if score_metric == 'lncc' or 'lmse':
                ndims = 3
                win = [win_len] * ndims
                # filter
                kp_filt = torch.ones([1, 1, *win]).to(device)
               
                # keypoint surrounding image
                kpt_img_fixed = np.zeros(win)
                fx_min, fx_max, fy_min, fy_max, fz_min, fz_max, fshiftx, fshifty, fshiftz = extract_range(kpt_f, win_len, img_fixed.shape)
                kpt_img_fixed = img_fixed[fx_min+fshiftx:fx_max+fshiftx+1, fy_min+fshifty:fy_max+fshifty+1, fz_min+fshiftz:fz_max+fshiftz+1]

            #         display(
            #             img_fixed, 
            #             slice_num=kpt_f,
            #             title=['fixed_img']*3
            #         )
                if verbose:
                    display(
                        keypoints_img(img_fixed, 
                                      kpt_fixed,
                                      kp_id=i)[0],
                        slice_num=kpt_f,
                        title=['fixed_kp']*3
                    )

                    display(
                        kpt_img_fixed,
                        title=['fixed_kp_win']*3
                    )

                kpt_img_moving = np.zeros(win)
                mx_min, mx_max, my_min, my_max, mz_min, mz_max, mshiftx, mshifty, mshiftz = extract_range(kpt_m, win_len, img_moving.shape)
                kpt_img_moving = img_moving[mx_min+mshiftx:mx_max+mshiftx+1, my_min+mshifty:my_max+mshifty+1, mz_min+mshiftz:mz_max+mshiftz+1]

            #         display(
            #             img_moving, 
            #             slice_num=kpt_m,
            #             title=['moving_img']*3
            #         )

                if verbose:
                    display(
                        keypoints_img(img_moving, 
                                      kpt_moving,
                                      kp_id=i)[0],
                        slice_num=kpt_m,
                        title=['moving_kp']*3
                    )

                    display(
                        kpt_img_moving,
                        title=['moving_kp_win']*3
                    )
                I = torch.from_numpy(kpt_img_fixed)[None, None, ...].to(device).to(torch.float32)
                J = torch.from_numpy(kpt_img_moving)[None, None, ...].to(device).to(torch.float32)
                weight = kp_filt
                conv_fn = F.conv3d

                # compute CC squares
                I2 = I * I
                J2 = J * J
                IJ = I * J

                # compute filters
                # compute local sums via convolution
                I_sum = conv_fn(I, weight, padding=int(win_len / 2))
                J_sum = conv_fn(J, weight, padding=int(win_len / 2))
                I2_sum = conv_fn(I2, weight, padding=int(win_len / 2))
                J2_sum = conv_fn(J2, weight, padding=int(win_len / 2))
                IJ_sum = conv_fn(IJ, weight, padding=int(win_len / 2))
                
                # compute cross correlation
                win_size = np.prod(win)
            elif score_metric == 'gncc' or 'gmse':
                I = torch.from_numpy(img_fixed)[None, None, ...].to(device).to(torch.float32)
                J = torch.from_numpy(img_moving)[None, None, ...].to(device).to(torch.float32)
                
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

            cc = cross * cross / (I_var * J_var + eps)
            ncc = cc.mean().item()
            #ncc = cc[cc > 0].mean().item()

            mse = torch.mean((I - J) ** 2).item()
        else:
            ncc = 0
            mse = 0
        
        tre = np.mean((kpt_f - kpt_m) ** 2)

        #print(f'keypoint {i} ncc: {ncc} mse: {mse} tre: {tre}')
        cc_score.append(ncc)
        mse_score.append(mse)
        tre_score.append(tre)
    
    return np.array(cc_score), np.array(mse_score), np.array(tre_score)


def __load_nii_img(img_path, preprocess: bool = False, downsample: int = 1, normalize: bool = True) -> np.ndarray:
    img = nib.load(img_path)
    arr = img.get_fdata()

    if preprocess:
        # clip pixel values
        min_bound = -1000.0
        max_bound = 500.0
        arr[arr < min_bound] = min_bound
        arr[arr > max_bound] = max_bound

        # normalize
        if normalize:
            arr = (arr - min_bound) / (max_bound - min_bound)

    if downsample != 1:
        arr = scipy.ndimage.zoom(arr, zoom=(1/downsample, 1/downsample, 1/downsample), order=0)


    return arr

def display(
        img, 
        kpt=None,
        box=None,
        slice_num=['mid','mid','mid'], 
        title=['view1', 'view2', 'view3'], 
        v_range=None, 
        cmap='gray',
        markerandcolor=['r.', 'r.', 'r.'],
        markersize=[10, 10, 10],
    ):
    '''
    kpt_fixed: (n, 3) 1 - 224, 1 - 192, 1 - 224
    box: (n, 6) [xl, yl, zl, xr, yr, zr]
    '''
    if v_range != None:
        v_min = v_range[0]
        v_max = v_range[1]
    else:
        v_min = img.min()
        v_max = img.max()
    if slice_num == ['mid', 'mid', 'mid']:
        slice_num = [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2]
    slice_num = [int(sn) for sn in slice_num]
    if isinstance(kpt, np.ndarray):
        kpt = np.round(kpt)
        pos0 = np.where(kpt[:, 0] == slice_num[0])[0]
        pos1 = np.where(kpt[:, 1] == slice_num[1])[0]
        pos2 = np.where(kpt[:, 2] == slice_num[2])[0]

    if isinstance(box, np.ndarray):
        box = np.round(box)
        print(box)
    plt.figure(figsize = (16, 16))
    plt.subplot(1, 3, 1)
    plt.imshow(img[slice_num[0]], cmap=cmap, vmin=v_min, vmax=v_max)
    if isinstance(kpt, np.ndarray):
        plt.plot(
            kpt[pos0, 2], 
            kpt[pos0, 1],
            markerandcolor[0], 
            markersize=markersize[0]
        )
    
    if isinstance(box, np.ndarray):
        bounding_box0 = patches.Rectangle(
            (box[pos0, 2], box[pos0, 1]), 
            box[pos0, 5] - box[pos0, 2],
            box[pos0, 4] - box[pos0, 1], 
            linewidth=1, 
            edgecolor='r',
            facecolor='none'
        )
        plt.gca().add_patch(bounding_box0)
    plt.title(title[0])

    plt.subplot(1, 3, 2)
    plt.imshow(img[:, slice_num[1]], cmap=cmap, vmin=v_min, vmax=v_max)
    if isinstance(kpt, np.ndarray):
        plt.plot(
            kpt[pos1, 2], 
            kpt[pos1, 0],
            markerandcolor[1], 
            markersize=markersize[1]
        )
    
    if isinstance(box, np.ndarray):
        bounding_box1 = patches.Rectangle(
            (box[pos1, 2], box[pos1, 0]), 
            box[pos1, 5] - box[pos1, 2],
            box[pos1, 3] - box[pos1, 0], 
            linewidth=1, 
            edgecolor='r',
            facecolor='none'
        )
        plt.gca().add_patch(bounding_box1)
    plt.title(title[1])

    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, slice_num[2]], cmap=cmap, vmin=v_min, vmax=v_max)
    if isinstance(kpt, np.ndarray):
        plt.plot(
            kpt[pos2, 1], 
            kpt[pos2, 0],
            markerandcolor[2], 
            markersize=markersize[2]
        )
    
    if isinstance(box, np.ndarray):
        bounding_box2 = patches.Rectangle(
            (box[pos2, 1], box[pos2, 0]), 
            box[pos2, 4] - box[pos2, 1],
            box[pos2, 3] - box[pos2, 0], 
            linewidth=1, 
            edgecolor='r',
            facecolor='none'
        )
        plt.gca().add_patch(bounding_box2)
    plt.title(title[2])

    plt.show()


def keypoints_img(img, keypoints, kp_id='all', r=1, color=[255, 0, 0]):
    #colormap = cm.get_cmap('viridis', 128)
    #p = np.linspace(0, 1, args.num_cluster + args.no_background)
    #color = np.array(colormap(0)) * 255.0
    kp_img = img[..., None]
    kp_img = np.tile(kp_img, reps=(1, 1, 1, 3))
    kp_only = np.zeros_like(img)
    if kp_id == 'all':
        for i, kp in enumerate(keypoints):
            if r == 0:
                kp_img[round(kp[0]), round(kp[1]), round(kp[2]), :] = color
                kp_only[round(kp[0]), round(kp[1]), round(kp[2])] = i + 1

            elif r > 0:
                kp_img[round(kp[0])-r:round(kp[0])+r,
                       round(kp[1])-r:round(kp[1])+r,
                       round(kp[2])-r:round(kp[2])+r, :] = color
                kp_only[round(kp[0])-r:round(kp[0])+r,
                        round(kp[1])-r:round(kp[1])+r,
                        round(kp[2])-r:round(kp[2])+r] = i + 1
    else:
        if r == 0:
            kp_img[round(keypoints[kp_id][0]),
                    round(keypoints[kp_id][1]),
                    round(keypoints[kp_id][2]), :] = color
            kp_only[round(keypoints[kp_id][0]),
                    round(keypoints[kp_id][1]),
                    round(keypoints[kp_id][2])] = kp_id + 1

        elif r > 0:
            kp_img[round(keypoints[kp_id][0])-r:round(keypoints[kp_id][0])+r,
                    round(keypoints[kp_id][1])-r:round(keypoints[kp_id][1])+r,
                    round(keypoints[kp_id][2])-r:round(keypoints[kp_id][2])+r, :] = color
            kp_only[round(keypoints[kp_id][0])-r:round(keypoints[kp_id][0])+r,
                    round(keypoints[kp_id][1])-r:round(keypoints[kp_id][1])+r,
                    round(keypoints[kp_id][2])-r:round(keypoints[kp_id][2])+r] = kp_id + 1

    return kp_img, kp_only

def kpimg2kp(kp_img):
    kp_img_flatten = kp_img.flatten() # h*w*d
    pos_kp_img_flatten = torch.where(kp_img_flatten > 0)[0] # pos(h*w*d) > 0
    index = kp_img_flatten[pos_kp_img_flatten].flatten()
    _, order = torch.topk(index, len(index), largest=False, sorted=True)
    pos_kp_img_flatten = pos_kp_img_flatten[order.to(torch.int64)]

    H, W, D = kp_img.shape[2:]
    [h, w, d] = torch.meshgrid(torch.arange(H), torch.arange(W), torch.arange(D))
    h, w, d = h.flatten(), w.flatten(), d.flatten()                
    kp = torch.zeros(kp_img.shape[0], kp_img.shape[1], int(kp_img.max().item()+1), 3) # b, c, n, 3 
    for count, pos in enumerate(pos_kp_img_flatten):
        hh, ww, dd = h[pos], w[pos], d[pos]
        kp[:, :, count] = torch.tensor([hh, ww, dd])

    return kp
