from utils import *
from models.layers import *


class CustomClip(torch.autograd.Function):
    """
    See https://interdigitalinc.github.io/CompressAI/_modules/compressai/ops/bound_ops.html#LowerBound
    """
    @staticmethod
    def forward(ctx, x, lower, upper):
        ctx.save_for_backward(x)
        ctx.lower = lower
        ctx.upper = upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        lower = ctx.lower
        upper = ctx.upper
        mask = ((x < lower) * (grad_output < 0)) + ((x > upper) * (grad_output > 0)) + (x >= lower) * (x <= upper)
        return mask * grad_output, None, None


def clip_with_grad(x, lower, upper):
    return CustomClip.apply(x, lower, upper)


def broadcast_prior(param, prior):
    return torch.ones_like(param) * prior


def params_to_vector(params):
    return nn.utils.parameters_to_vector([v for k, v in sorted(params.items())])


def priors_to_vector(params, priors):
    return nn.utils.parameters_to_vector([broadcast_prior(v, priors[k]) for k, v in sorted(params.items())])


def vector_to_params(params, param_vec):
    params = {k: torch.clone(v.detach()) for k, v in sorted(params.items())}
    nn.utils.vector_to_parameters(param_vec, [v for k, v in sorted(params.items())])
    return params


def num_bits(x):
    assert isinstance(x, torch.Tensor), 'Inputs should be tensors.'
    return x.numel() * x.dtype.itemsize * 8


def forward_meta(x_in):
    rates = {}
    x_out = {}
    for k, v in x_in.items():
        assert v is None or isinstance(v, torch.Tensor), 'Inputs should be tensors.'
        if v is None:
            continue
        rates[k] = num_bits(v)
        x_out[k] = v
    return rates, x_out


def compress_meta(bs, x_in):
    with torch.no_grad():
        rates = {}
        x_out = {}
        for k, v in sorted(x_in.items()):
            assert v is None or isinstance(v, torch.Tensor), 'Inputs should be tensors.'
            if v is None:
                continue

            # Convert to numpy
            v_np = v.cpu().numpy() 

            # Encode bitstream
            v_bs = v_np.tobytes()
            bs += v_bs
            rates[k] = len(v_bs) * 8

            # Recover the original tensor
            v_dec = torch.tensor(np.frombuffer(v_bs, dtype=v_np.dtype), device=v.device).view_as(v)
            x_out[k] = v_dec

        return bs, rates, x_out


def decompress_meta(bs, x_temp):
    with torch.no_grad():
        rates = {}
        x_out = {}
        for k, v in sorted(x_temp.items()):
            assert v is None or isinstance(v, torch.Tensor), 'Inputs should be tensors.'
            if v is None:
                continue

            # Convert to numpy
            v_np = v.cpu().numpy()
    
            # Decode bitstream
            v_bs, bs = bs[:v_np.nbytes], bs[v_np.nbytes:]
            rates[k] = len(v_bs) * 8

            # Recover the original tensor
            v_dec = torch.tensor(np.frombuffer(v_bs, dtype=v_np.dtype), device=v.device).view_as(v)
            x_out[k] = v_dec

        return bs, rates, x_out


def masked_mean(x, mask, dim=None, keepdim=False, eps=1e-6):
    assert mask is None or mask.dtype == torch.bool
    x = x.float()
    mask = mask.float() if mask is not None else torch.ones_like(x)
    x_mask = x * mask
    N = mask.sum(dim, keepdim=keepdim)
    mean = x_mask.sum(dim, keepdim=keepdim) / (N + eps)
    return mean


def masked_std(x, mask, correction=1, dim=None, keepdim=False, eps=1e-6):
    assert mask is None or mask.dtype == torch.bool
    x = x.float()
    mask = mask.float() if mask is not None else torch.ones_like(x)
    x_mask = x * mask
    N = mask.sum(dim, keepdim=True)
    mean = x_mask.sum(dim, keepdim=True) / (N + eps)
    std = (((x_mask - mean) ** 2 / (torch.clip(N - correction, min=0) + eps)).sum(dim, keepdim=keepdim)).sqrt()
    return std


def init_log_step(shape, value=-6.):
    return create_param(torch.full(shape, value), 'em_log-step')


def set_width(x, log_step, n, dim=None):
    if n is None:
        return
    x_range = x.amax(dim=dim, keepdim=True) - x.amin(dim=dim, keepdim=True)
    step = x_range / (2. ** n - 1.)
    log_step.copy_(torch.clip(torch.log(step), -6., -2.).expand_as(log_step))


def scale(x, log_step):
    if not isinstance(log_step, torch.Tensor):
        log_step = torch.tensor(log_step, device=x.device)
    step = torch.exp(clip_with_grad(log_step, -10., 0.))
    return x / step


def unscale(x, log_step):
    if not isinstance(log_step, torch.Tensor):
        log_step = torch.tensor(log_step, device=x.device)
    step = torch.exp(clip_with_grad(log_step, -10., 0.))
    return x * step


def init_means(shape, means=0.):
    return create_param(torch.full(shape, means), 'em_mean')


def init_log_scales(shape, scales=1.):
    return create_param(torch.log(torch.full(shape, scales)),'em_log-scale')


def set_means(x, means):
    means.copy_(x.mean().expand_as(means))


def set_log_scales(x, log_scales):
    log_scales.copy_(torch.zeros_like(log_scales))


def is_shape_compatible(x1, x2):
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor), 'Inputs should be tensors.'
    if len(x1.shape) != len(x2.shape):
        return False
    for i in range(len(x1.shape)):
        if x1.shape[i] != x2.shape[i] and x1.shape[i] != 1 and x2.shape[i] != 1:
            return False
    return True


def is_shape_the_same(x1, x2):
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor), 'Inputs should be tensors.'
    return x1.shape == x2.shape