"""
Model utils
"""
from utils import *


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def cfg_override(cfg, **kwargs):
    cfg = cfg.copy()
    for k, v in kwargs.items():
        cfg[k] = v
    return cfg


def assert_shape(x: torch.Tensor, shape: tuple):
    assert tuple(x.shape) == tuple(shape), f'shape: {x.shape}     expected: {shape}'


def compute_multiscale_sizes(size, scales, divisible_only=False):
    sizes = [size]
    for scale in scales:
        assert not divisible_only or all(size[d] % scale[d] == 0 for d in range(len(size)))
        size = tuple(int(size[d] / scale[d]) for d in range(len(size)))
        sizes.append(size)
    return sizes