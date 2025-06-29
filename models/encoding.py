"""
Encoding
"""
from .utils import *
from .layers import *
from .patch_utils import *
from .layers import *


def _compute_normalized_coordinate(l: torch.IntTensor, L: int, align_corners: bool):
    if L == 1.:
        return l * 0.
    elif align_corners:
        step = 2. / (L - 1)
        return -1. + l * step
    else:
        step = 2. / L
        return -1. + step / 2. + l * step


def _compute_normalized_coordinates_3d(l: List[torch.IntTensor], L: List[int], align_corners: bool):
    return [_compute_normalized_coordinate(l[d], L[d], align_corners) for d in range(3)]


def _interpolate3D(grid: torch.Tensor, coor_wht: torch.Tensor, align_corners: bool=False):
    _, _, _, C = grid.shape
    N, T, H, W, _ = coor_wht.shape
    # [T_grid, H_grid, W_grid, C] -> [1, C, T_grid, H_grid, W_grid]
    grid = grid.permute(3, 0, 1, 2).unsqueeze(0)
    # [N, T, H, W, 3] -> [1, N * T, H, W, 3].
    coor_wht = coor_wht.view(1, N * T, H, W, 3)
    # Always use fp32 for grid_sample
    grid = grid.float()
    coor_wht = coor_wht.float()
    return F.grid_sample(grid, coor_wht, mode='bilinear', padding_mode='border', align_corners=align_corners) \
            .view(C, N, T, H, W).permute(1, 2, 3, 4, 0).contiguous()


def _compute_grid_encoding(grids: List[torch.FloatTensor], coors: List[torch.FloatTensor],
                           align_corners: bool=False, reduce='concat'):
    assert coors[0].ndim == 2 and (coors[1] is None or coors[1].ndim == 2) and (coors[2] is None or coors[2].ndim == 2)
    assert (coors[1] is None) == (coors[2] is None)
    N, T = coors[0].shape
    H = coors[1].shape[1] if coors[1] is not None else 1
    W = coors[2].shape[1] if coors[2] is not None else 1

    # Sampling coordinates
    if H > 1 and W > 1:
        coor_t = coors[0][:, :, None, None, None].expand(N, T, H, W, 1)
        coor_h = coors[1][:, None, :, None, None].expand(N, T, H, W, 1)
        coor_w = coors[2][:, None, None, :, None].expand(N, T, H, W, 1)
        coor_wht = torch.cat([coor_w, coor_h, coor_t], dim=-1) # (x, y, t)/(w, h, t) order
    else:
        coor_t = coors[0].view(N, T, 1, 1, 1)
        coor_wht = F.pad(coor_t, (2, 0)) # (x, y, t)/(w, h, t) order

    enc = []

    for grid_i in grids:
        # Interpolate in all dimenions
        enc_i = _interpolate3D(grid=grid_i.view(*grid_i.shape[:3], -1), coor_wht=coor_wht,
                               align_corners=align_corners) \
                              .view(N, T, H, W, *grid_i.shape[3:])
        enc.append(enc_i)

    if reduce == 'concat':
        return torch.concat(enc, dim=-1)
    elif reduce == 'sum':
        return sum(enc)
    elif reduce == 'none':
        return enc
    else:
        raise ValueError


class BaseEncodingImpV1(FuncModule):
    """
    Base Encoding.
    """
    def __init__(self, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.paddings = paddings
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.kernel_size = kernel_size
        self.grid_expands = nn.ModuleList()
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.func_params_keys = [f'grids_{i}' for i in range(self.grid_level)]

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = ((int(grid_size[d] / (self.grid_level_scale[d] ** i)) \
                                                       for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, C_grid_i))
            self.grid_expands.append(BatchGridTrilinear3D(self.out_size, self.align_corners))

        if self.kernel_size[0] > 1:
            raise NotImplementedError
        elif self.kernel_size[1] > 1 or self.kernel_size[2] > 1:
            self.proj = FuncConv2D(sum(grid_size_i[-1] for grid_size_i in self.grid_sizes),
                                   self.out_channels, self.kernel_size[1:])
        else:
            self.proj = FuncLinear(sum(grid_size_i[-1] for grid_size_i in self.grid_sizes), self.out_channels)

    def extra_repr(self):
        s = 'out_size={out_size}, out_channels={out_channels}, paddings={paddings}, grid_sizes={grid_sizes}, align_corners={align_corners}'
        return s.format(**self.__dict__)

    def _get_params(self):
        params = {}
        for i, grid_size_i in enumerate(self.grid_sizes):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = grid_size_i
            grid_i = create_param(torch.zeros(T_grid_i, H_grid_i, W_grid_i, C_grid_i), 'grid')
            torch.nn.init.uniform_(grid_i, -self.init_scale, self.init_scale)
            params[f'grids_{i}'] = grid_i
        return params

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool=True):
        grids = self._extract_params(params)

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # Compute the global voxels coordinates        
        padding = self.paddings if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        px_idx_flat = (px_idx[0][:, :, None, None] * self.out_size[1] * self.out_size[2] + \
                       px_idx[1][:, None, :, None] * self.out_size[2] + \
                       px_idx[2][:, None, None, :]).view(-1)

        # Interpolate grids
        grids_upsampled = torch.concat([self.grid_expands[i](grids[i]) for i in range(self.grid_level)], dim=-1)
        grids_upsampled_flat = grids_upsampled.view((-1, grids_upsampled.shape[-1],))

        # Extract encodings
        encoding = torch.index_select(grids_upsampled_flat, 0, px_idx_flat) \
                        .view((idx.shape[0],) + patch_padded_size + (grids_upsampled.shape[-1],))
        output = px_mask_3d * self.proj(px_mask_3d * encoding, params)
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        return output


class BaseEncodingImpV2(FuncModule):
    """
    Base Encoding V2
    """
    def __init__(self, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.out_channels = out_channels
        self.paddings = paddings
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.kernel_size = kernel_size
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.func_params_keys = [f'grids_{i}' for i in range(self.grid_level)]

        for i in range(self.grid_level):
            T_grid_i, H_grid_i, W_grid_i, C_grid_i = ((int(grid_size[d] / (self.grid_level_scale[d] ** i)) \
                                                       for d in range(4)))
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, C_grid_i))
            if self.kernel_size[0] > 1:
                self.add_module(f'proj_{i}', FuncConv3D(C_grid_i, self.out_channels, self.kernel_size))
            else:
                self.add_module(f'proj_{i}', FuncConv2D(C_grid_i, self.out_channels, self.kernel_size[1:]))

    def extra_repr(self):
        s = 'out_size={out_size}, out_channels={out_channels}, paddings={paddings}, grid_sizes={grid_sizes} align_corners={align_corners}'
        return s.format(**self.__dict__)

    def _get_params(self):
        params = {}
        for i, grid_size_i in enumerate(self.grid_sizes):
            grid_i = create_param(torch.zeros(grid_size_i), 'grid')
            torch.nn.init.uniform_(grid_i, -self.init_scale, self.init_scale)
            params[f'grids_{i}'] = grid_i
        return params

    def _forward_proj_first(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                            patch_mode: bool=True):
        grids = self._extract_params(params)
        grids_projected = [getattr(self, f'proj_{i}')(grids[i].unsqueeze(0), params).squeeze(0) \
                           for i in range(self.grid_level)]

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # Compute the global voxels coordinates
        padding = self.paddings if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        coor = _compute_normalized_coordinates_3d(px_idx, self.out_size, self.align_corners)

        # Extract encodings
        encoding = _compute_grid_encoding(grids_projected, coor, self.align_corners, reduce='sum')
        output = px_mask_3d * encoding
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        return output

    def _forward_interp_first(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                              patch_mode: bool=True):
        grids = self._extract_params(params)
        patch_mode = True # Always set patch_mode to True - this ensures the border pixel is used for padding

        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        # Compute the global voxels coordinates
        padding = tuple(self.paddings[d] + (self.kernel_size[d] - 1) // 2 for d in range(3)) \
                  if patch_mode else (0, 0, 0)
        patch_size = tuple(self.out_size[d] // idx_max[d] for d in range(3))
        patch_padded_size = tuple(patch_size[d] + 2 * padding[d] for d in range(3))

        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, self.out_size, padding=padding, clipped=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        coor = _compute_normalized_coordinates_3d(px_idx, self.out_size, self.align_corners)

        # Extract encodings
        encoding = _compute_grid_encoding(grids, coor, self.align_corners, reduce='none')
        output = px_mask_3d * sum(getattr(self, f'proj_{i}')(encoding[i], params) for i in range(self.grid_level))
        assert_shape(output, (idx.shape[0],) + patch_padded_size + (self.out_channels,))

        if patch_mode:
            if self.kernel_size[0] > 1:
                output = output[:, (self.kernel_size[0] - 1) // 2:-(self.kernel_size[0] - 1) // 2, :, :, :]
            if self.kernel_size[1] > 1:
                output = output[:, :, (self.kernel_size[1] - 1) // 2:-(self.kernel_size[1] - 1) // 2, :, :]
            if self.kernel_size[2] > 1:
                output = output[:, :, :, (self.kernel_size[2] - 1) // 2:-(self.kernel_size[2] - 1) // 2]
            output = output.contiguous()

        return output

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool=True):
        # If grid size is the same as the output size, perform interpolation first (results are the same but faster).
        # Otherwise, perform projection first (results are different if perform interpolation first)
        interp_first = all(all(grid_size_i[:-1][d] == self.out_size[d] for d in range(3)) \
                           for grid_size_i in self.grid_sizes)
        if interp_first:
            return self._forward_interp_first(params, idx, idx_max, patch_mode)
        else:
            return self._forward_proj_first(params, idx, idx_max, patch_mode)


class BaseEncoding(FuncModule):
    def __init__(self, impl, out_size, out_channels, paddings: tuple=(0, 0, 0),
                 grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5], kernel_size=(1, 1, 1),
                 init_scale=1e-3, align_corners=False):
        super().__init__()
        assert isinstance(out_size, (list, tuple)) and len(out_size) == 3
        self.out_size = out_size
        self.out_channels = out_channels
        if impl == 'v1':
            grid_cls = BaseEncodingImpV1
        elif impl == 'v2':
            grid_cls = BaseEncodingImpV2
        else:
            raise ValueError
        self.enc = grid_cls(out_size, out_channels, paddings, grid_size, grid_level, grid_level_scale, kernel_size, init_scale, align_corners)

    def forward(self, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int], patch_mode: bool=True):
        return getattr(self, f'enc')(params, idx, idx_max, patch_mode)


def create_base_encoding(logger, grid_type, **kwargs):
    if grid_type in ['v1', 'v2']:
        return BaseEncoding(grid_type, **kwargs)
    else:
        raise ValueError