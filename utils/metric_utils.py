import numpy as np
import scipy.ndimage
from scipy.ndimage import map_coordinates

"""
Metrics fucntions adopted from https://github.com/MDL-UzL/L2R/blob/main/evaluation
"""


##### metrics #####
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )
    
    jacdet = (jacdet + 3).clip(1e-9, 1e+9)
    return (jacdet <= 0).astype(float).sum(), np.log(jacdet)


def compute_tre(fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array(
        (fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)
    ).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp
    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=2).mean(1)

def compute_all_tre(fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array(
        (fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)
    ).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp
    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=2) #.mean(1)

def compute_dice_coefficient(mask_gt, mask_pred):
    """Computes soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return 0
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_dice(fixed, moving, warped, labels, reg_dir='f2m'):
    dice = []
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((moving == i), (warped == i)))
    mean_dice = np.nanmean(dice)
    return mean_dice, dice
