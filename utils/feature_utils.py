import torch
import torch.nn as nn
import torch.nn.functional as F

# ### feature #####

def mindssc(img, delta=1, sigma=0.8):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    
    device = img.device
    dtype = img.dtype
    
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()
    
    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)
    
    # compute patch-ssd
    ssd = smooth(((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2), sigma)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)
    
    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    
    return mind

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    return img

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)



# Keypoint Sampling
#def keypoint_sampling(kpts_fixed, kpts_mov, img_fixed, img_mov, sampling_method="random"):
#    if sampling_method == "random":
#        raise NotImplementedError
#
#    elif sampling_method == "ncc":
        
