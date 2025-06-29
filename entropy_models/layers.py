from .utils import *


class FuncMaskedConv2d(FuncConv2D):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool = True,
                 mask_cur: bool = True, is_head: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur
        self.is_head = is_head

    def _get_params(self):
        param_prefix = 'em_conv' if self.is_head else 'conv'
        weight_shape = (self.out_channels, self.in_channels // self.groups,
                        math.prod(self.kernel_size) // 2 + (not self.mask_cur))
        weight = create_param(torch.empty(weight_shape), f'{param_prefix}_weight')
        bias = create_param(torch.zeros(self.out_channels), f'{param_prefix}_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        weight = F.pad(weight, (0, math.prod(self.kernel_size) - weight.shape[-1])) \
                  .view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == '' else self.func_key + '__'
        return super().forward(x, {f'{prefix}weight': weight, f'{prefix}bias': bias})


class FuncMaskedConv3d(FuncConv3D):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: str = 'same', dilation: int = 1, groups: int = 1, bias: bool = True,
                 mask_cur: bool = True, is_head: bool = False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_shape = (1, 1, *self.kernel_size)
        self.mask_cur = mask_cur
        self.is_head = is_head

    def _get_params(self):
        param_prefix = 'em_conv' if self.is_head else 'conv'
        weight_shape = (self.out_channels, self.in_channels // self.groups, 
                        math.prod(self.kernel_size) // 2 + (not self.mask_cur))
        weight = create_param(torch.empty(weight_shape), f'{param_prefix}_weight')
        bias = create_param(torch.zeros(self.out_channels), f'{param_prefix}_bias') if self.bias else None
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return {'weight': weight, 'bias': bias}

    def forward(self, x, params):
        assert x.ndim == 5, 'x should have 5 dimensions with shape [N, T, H, W, C]'
        weight, bias = self._extract_params(params)
        weight = F.pad(weight, (0, math.prod(self.kernel_size) - weight.shape[-1])) \
                  .view(*weight.shape[:-1], *self.kernel_size)
        prefix = self.func_key if self.func_key == '' else self.func_key + '__'
        return super().forward(x, {f'{prefix}weight': weight, f'{prefix}bias': bias})