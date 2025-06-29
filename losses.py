"""
Losses
"""
from utils import *
import pytorch_msssim


def check_shape(x, y):
    assert x.shape == y.shape, 'shape of tensors must be the same!'
    assert x.ndim == y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'


def yuv444to420(x):
    assert x.shape[3] % 2 == x.shape[4] % 2 == 0, 'height and width must be even'
    assert x.shape[1] == 3, 'inputs are expected to have 3 channels'
    N, C, T, H, W = x.shape
    x_down = F.avg_pool2d(x.view(N, C * T, H, W), kernel_size=2, stride=2).view(N, C, T, H // 2, W // 2)
    return x[:, 0:1], x_down[:, 1:2], x_down[:, 2:3]


def mse(x, y):
    """
    Compute the per-frame MSE loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction='none').mean(dim=2)


def log_mse(x, y):
    """
    Compute the per-frame log MSE loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return torch.log(F.mse_loss(x, y, reduction='none')).mean(dim=2)


def mse_yuv611_sum(x, y):
    """
    Compute the per-frame MSE loss
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    check_shape(x, y)
    N, _, T, _, _ = x.shape
    x = [x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for x_i in yuv444to420(x)]
    y = [y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for y_i in yuv444to420(y)]
    yuv_weight = torch.tensor([6. / 8., 1. / 8., 1. / 8.], device=x[0].device).view(1, 1, 3)
    yuv_loss = torch.concat([F.mse_loss(x_i, y_i, reduction='none').mean(dim=2) for x_i, y_i in zip(x, y)], dim=2)
    return (yuv_weight * yuv_loss).sum(dim=2)


def mse_yuv611_product(x, y):
    """
    Compute the per-frame MSE loss
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    check_shape(x, y)
    N, _, T, _, _ = x.shape
    x = [x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for x_i in yuv444to420(x)]
    y = [y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for y_i in yuv444to420(y)]
    yuv_weight = torch.tensor([6. / 8., 1. / 8., 1. / 8.], device=x[0].device).view(1, 1, 3)
    yuv_loss = torch.concat([F.mse_loss(x_i, y_i, reduction='none').mean(dim=2) for x_i, y_i in zip(x, y)], dim=2)
    return (yuv_loss ** yuv_weight).prod(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction='none').mean(dim=2)


def l1_yuv611(x, y):
    """
    Compute the per-frame L1 loss
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    check_shape(x, y)
    N, _, T, _, _ = x.shape
    x = [x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for x_i in yuv444to420(x)]
    y = [y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for y_i in yuv444to420(y)]
    yuv_weight = torch.tensor([6. / 8., 1. / 8., 1. / 8.], device=x[0].device).view(1, 1, 3)
    yuv_loss = torch.concat([F.l1_loss(x_i, y_i, reduction='none').mean(dim=2) for x_i, y_i in zip(x, y)], dim=2)
    return (yuv_weight * yuv_loss).sum(dim=2)


def psnr(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    """
    return 10 * torch.log10((v_max ** 2) / (mse(x, y) + 1e-9))


def psnr_yuv611(x, y, v_max=1.):
    """
    Compute the per-frame PSNR
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    check_shape(x, y)
    N, _, T, _, _ = x.shape
    x = [x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for x_i in yuv444to420(x)]
    y = [y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1) for y_i in yuv444to420(y)]
    yuv_weight = torch.tensor([6. / 8., 1. / 8., 1. / 8.], device=x[0].device).view(1, 1, 3)
    yuv_mse = torch.concat([F.mse_loss(x_i, y_i, reduction='none').mean(dim=2) for x_i, y_i in zip(x, y)], dim=2)
    return ((10 * torch.log10((v_max ** 2) / (yuv_mse + 1e-9))) * yuv_weight).sum(dim=2)


def _create_ssim_win(x, size, sigma):
    """
    Create 1-D Gaussian kernel on the target device
    Modified from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    coords = torch.arange(size, dtype=torch.float, device=x.device)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0).repeat([x.shape[1]] + [1] * (len(x.shape) - 1))


def ssim(x, y, v_max=1., win_size=11, win_sigma=1.5):
    """
    Compute the per-frame SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    win = _create_ssim_win(x, win_size, 1.5)
    return pytorch_msssim.ssim(x, y, v_max, win_size=win_size, win_sigma=win_sigma,
                               win=win, size_average=False).view(N, T)


def ms_ssim(x, y, v_max=1., win_size=11, win_sigma=1.5):
    """
    Compute the per-frame MS-SSIM
    """
    check_shape(x, y)
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    win = _create_ssim_win(x, win_size, 1.5)
    return pytorch_msssim.ms_ssim(x, y, v_max, win_size=win_size, win_sigma=win_sigma,
                                  win=win, size_average=False).view(N, T)


def ssim_y(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame SSIM for Y channel
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    return ssim(x[:, 0:1], y[:, 0:1], v_max, win_size)


def ms_ssim_y(x, y, v_max=1., win_size=11):
    """
    Compute the per-frame MS-SSIM for Y channel
    """
    assert x.shape[1] == y.shape[1] == 3, 'inputs are expected to have 3 channels'
    return ms_ssim(x[:, 0:1], y[:, 0:1], v_max, win_size)


def compute_loss(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()

    if name == 'mse':
        loss = mse(x, y)
    elif name == 'log-mse':
        loss = log_mse(x, y)
    elif name == 'mse-yuv611-s':
        loss = mse_yuv611_sum(x, y)
    elif name == 'mse-yuv611-p':
        loss = mse_yuv611_product(x, y)
    elif name == 'l1-yuv611':
        loss = l1_yuv611(x, y)
    elif name == 'l1':
        loss = l1(x, y)
    elif name == 'ssim':
        loss = 1. - ssim(x, y)
    elif name == 'ssim-5x5':
        loss = 1. - ssim(x, y, win_size=5)
    elif name == 'ms-ssim':
        loss = 1. - ms_ssim(x, y)
    elif name == 'ms-ssim-5x5':
        loss = 1. - ms_ssim(x, y, win_size=5)
    elif name == 'ssim-y':
        loss = 1. - ssim_y(x, y)
    elif name == 'ssim-y-5x5':
        loss = 1. - ssim_y(x, y, win_size=5)
    elif name == 'ms-ssim-y':
        loss = 1. - ms_ssim_y(x, y)
    elif name == 'ms-ssim-y-5x5':
        loss = 1. - ms_ssim_y(x, y, win_size=5)
    else:
        raise ValueError
    assert loss.ndim == 2, 'loss is expected to have 2D ([N, T])'
    return loss


def compute_metric(name, x, y):
    assert x.ndim == 5 and y.ndim == 5, 'inputs are expected to have 5D ([N, C, T, H, W])'
    x, y = x.float(), y.float()

    if name == 'mse':
        metric = mse(x, y)
    elif name == 'l1':
        metric = l1(x, y)
    elif name == 'psnr':
        metric = psnr(x, y)
    elif name == 'psnr-yuv611':
        metric = psnr_yuv611(x, y)
    elif name == 'ssim':
        metric = ssim(x, y)
    elif name == 'ssim-5x5':
        metric = ssim(x, y, win_size=5)
    elif name == 'ms-ssim':
        metric = ms_ssim(x, y)
    elif name == 'ms-ssim-5x5':
        metric = ms_ssim(x, y, win_size=5)
    elif name == 'ssim-y':
        metric = ssim_y(x, y)
    elif name == 'ssim-y-5x5':
        metric = ssim_y(x, y, win_size=5)
    elif name == 'ms-ssim-y':
        metric = ms_ssim_y(x, y)
    elif name == 'ms-ssim-y-5x5':
        metric = ms_ssim_y(x, y, win_size=5)
    else:
        raise ValueError
    assert metric.ndim == 2, 'metric is expected to have 2D ([N, T])'
    return metric