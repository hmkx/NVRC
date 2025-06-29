"""
HiNeRV
"""
from .utils import *
from .layers import *
from .upsample import FastTrilinearInterpolation, FastNearestInterpolation, crop_tensor_nthwc
from .patch_utils import *
from .encoding import create_base_encoding, _compute_normalized_coordinate, _compute_grid_encoding


@dataclass
class BaseEncodingConfig:
    base_grid_type: str = 'v1'
    base_grid_size: list[int] = field(default_factory=lambda: [-1, -1, -1, 2])
    base_grid_level: int = 2
    base_grid_level_scale: list[float] = field(default_factory=lambda: [2., 1., 1., .5])
    base_kernel: list[int] = field(default_factory=lambda: [1, 3, 3])


@dataclass
class HiNeRVDecoderConfig:
    # Network
    channels: list[int] = field(default_factory=lambda: [280, 140, 70, 35])
    depths: list[int] = field(default_factory=lambda: [3, 3, 3, 1])
    exps: list[float] = field(default_factory=lambda: [4., 4., 4., 1.])
    kernels: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    scales_t: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    scales_hw: list[int] = field(default_factory=lambda: [5, 3, 2, 2])
    paddings: list[int] = field(default_factory=lambda: [-1, -1, -1])

    # Block
    block_type: list[str] = field(default_factory=lambda: ['convnext'] * 4)
    block_norm: list[str] = field(default_factory=lambda: ['layernorm-no-affine'] * 4)
    block_act: list[str] = field(default_factory=lambda: ['gelu'] * 4)
    block_bias: bool = False
    block_layerscale: float = 0.0

    # Head
    head_type: str = 'linear_head'
    head_norm: str = 'none'
    head_act: str = 'sigmoid'
    head_bias: bool = True
    head_layerscale: float = 0.0

    # Upsample
    upsample_type: list[str] = field(default_factory=lambda: ['trilinear'] * 4)
    upsample_config: list[str] = field(default_factory=lambda: ['matmul-th-w'] * 4)
    upsample_norm: list[str] = field(default_factory=lambda: ['layernorm-no-affine'] * 4)
    upsample_act: list[str] = field(default_factory=lambda: ['none'] * 4)

    # Hierarchical Encoding
    use_hi_encoding: list[bool] = field(default_factory=lambda: [True] * 4)
    local_grid_size: list[int] = field(default_factory=lambda: [-1, 4])
    local_grid_level: int = 3
    local_grid_level_scale: list[float] = field(default_factory=lambda: [2., 0.5])
    local_grid_depth_scale: list[float] = field(default_factory=lambda: [1., 0.5])


@dataclass
class HiNeRVConfig:
    base_encoding: BaseEncodingConfig = field(default_factory=BaseEncodingConfig)
    decoder: HiNeRVDecoderConfig = field(default_factory=HiNeRVDecoderConfig)
    eval_patch_size: list[int] = field(default_factory=lambda: None)

    def __post_init__(self):
        self.base_encoding = BaseEncodingConfig(**self.base_encoding) \
                             if isinstance(self.base_encoding, dict) else self.base_encoding
        self.decoder = HiNeRVDecoderConfig(**self.decoder) \
                       if isinstance(self.decoder, dict) else self.decoder


def get_hi_encoding_cfg(out_channels, depth, size, scale, **kwargs):
    # Grid
    if len(kwargs['local_grid_size']) == 2:
        # Local learned (temporal only)
        T_grid, C_grid = kwargs['local_grid_size']
        H_grid, W_grid = 1, 1
        if 'local_grid_depth_scale' in kwargs:
            T_grid_scale, C_grid_scale = kwargs['local_grid_depth_scale']
            H_grid_scale, W_grid_scale = 1, 1
        else:
            T_grid_scale = H_grid_scale = W_grid_scale = C_grid_scale = 1.
    else:
        raise NotImplementedError

    T_grid = max(int(T_grid * T_grid_scale ** depth), 1) if T_grid != -1 else size[0]
    H_grid = max(int(H_grid * H_grid_scale ** depth), 1) if H_grid != -1 else size[1]
    W_grid = max(int(W_grid * W_grid_scale ** depth), 1) if W_grid != -1 else size[2]
    C_grid = max(int(C_grid * C_grid_scale ** depth), 1) if C_grid != -1 else size[3]

    if len(kwargs['local_grid_size']) == 4:
        kwargs['local_grid_size'] = [T_grid, H_grid, W_grid, C_grid]
    elif len(kwargs['local_grid_size']) == 2:
        kwargs['local_grid_size'] = [T_grid, C_grid]
    else:
        raise NotImplementedError

    # All configs
    hi_enc_cfg = {
        # grid encoding
        'out_channels': out_channels, 'grid_size': kwargs['local_grid_size'],
        'grid_level': kwargs['local_grid_level'],
        'grid_level_scale': kwargs['local_grid_level_scale'],
        'grid_stride': scale
    }

    return hi_enc_cfg


class HierarchicalEncoding(FuncModule):
    """
    Hierarchical encoding
    """
    def __init__(self, out_channels=128, grid_size=[120, 9, 16, 64], grid_level=3, grid_level_scale=[2., 1., 1., .5],
                 grid_stride=(1, 3, 3), init_scale=1e-3, align_corners=False):
        super().__init__()
        self.out_channels = out_channels
        self.grid_sizes = []
        self.grid_level = grid_level
        self.grid_level_scale = grid_level_scale
        self.grid_stride = grid_stride
        self.init_scale = init_scale
        self.align_corners = align_corners
        self.func_params_keys = [f'grids_{i}' for i in range(self.grid_level)]

        for i in range(self.grid_level):
            if len(grid_size) == 4:
                T_grid_i, H_grid_i, W_grid_i, C_grid_i = ((max(int(grid_size[d] / (self.grid_level_scale[d] ** i), 1)) \
                                                           for d in range(4)))
            elif len(grid_size) == 2:
                T_grid_i, C_grid_i = ((max(int(grid_size[d] / (self.grid_level_scale[d] ** i)), 1) for d in range(2)))
                H_grid_i, W_grid_i = 1, 1
            else:
                raise NotImplementedError
            self.grid_sizes.append((T_grid_i, H_grid_i, W_grid_i, *self.grid_stride, C_grid_i))

        self.linear = FuncLinear(sum(grid_size_i[-1] for grid_size_i in self.grid_sizes), self.out_channels)

    def extra_repr(self):
        s = 'out_channels={out_channels}, grid_level={grid_level}, grid_sizes={grid_sizes}, grid_stride={grid_stride}'
        return s.format(**self.__dict__)

    def _get_params(self):
        params = {}
        for i, grid_size_i in enumerate(self.grid_sizes):
            grid_i = create_param(torch.zeros(grid_size_i), 'local_grid')
            torch.nn.init.uniform_(grid_i, -self.init_scale, self.init_scale)
            params[f'grids_{i}'] = grid_i
        return params

    def forward(self, x: torch.Tensor, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int]):
        grids = self._extract_params(params)
        N, T, H, W, C = x.shape

        # This implementation is only correct with no temporal upsampling
        assert scale[0] == 1

        # Compute the global voxel coordinates before upscaling
        pre_size = (size[0] // scale[0],)
        pre_padding = (int(math.ceil(padding[0] / scale[0])),)
        pre_idx = compute_pixel_idx_1d(idx[:, 0:1], idx_max[0:1], pre_size, pre_padding,
                                       clipped=False, return_mask=False)
        coor_t = _compute_normalized_coordinate(pre_idx[0], pre_size[0], self.align_corners)

        # Compute the local voxel indexes
        px_idx, px_mask = compute_pixel_idx_3d(idx, idx_max, size, padding, clipped=False, return_mask=True)
        px_mask_3d = px_mask[0][:, :, None, None, None] * \
                     px_mask[1][:, None, :, None, None] * \
                     px_mask[2][:, None, None, :, None]
        lpx_idx = tuple(px_idx[d] % scale[d] for d in range(3))

        # Compute the encoding indexes
        enc_idx = tuple(torch.arange(scale[d], device=x.device) for d in range(3))

        # Encoding
        M = tuple(lpx_idx[d][:, :, None] == enc_idx[d][None, None, :] for d in range(3))
        M_3d = M[0][:, :, None, None, :, None, None] * \
               M[1][:, None, :, None, None, :, None] * \
               M[2][:, None, None, :, None, None, :]
        local_enc = self.linear(_compute_grid_encoding(grids, [coor_t, None, None],
                                                       self.align_corners, reduce='concat'),
                                params)
        local_enc_masked = px_mask_3d * \
                           torch.matmul(M_3d.view(N, T, H * W, scale[0] * scale[1] * scale[2]).to(local_enc.dtype),
                                        local_enc.view(N, T, scale[0] * scale[1] * scale[2], C)).view_as(x)
        return x + local_enc_masked
    

class Upsample(FuncModule):
    """
    HiNeRV Upsample layer.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.channels = kwargs['channels']
        self.scale = kwargs['scale']
        self.type = kwargs['type']
        self.norm = get_func_norm(kwargs['norm'])(self.channels)
        self.act = get_func_activation(kwargs['act'])()

        # Layer
        if self.type == 'trilinear':
            self.layer = FastTrilinearInterpolation(kwargs['config'])
        elif self.type == 'nearest':
            self.layer = FastNearestInterpolation(kwargs['config'])
        else:
            raise ValueError

    def extra_repr(self):
        s = 'scale={scale}, type={type}'
        return s.format(**self.__dict__)

    def _get_params(self):
        return {}

    def forward(self, x: torch.Tensor, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int],
                size: tuple[int, int, int], scale: tuple[int, int, int], padding: tuple[int, int, int], patch_mode: bool=True,
                mask: Optional[torch.Tensor]=None):
        assert x.ndim == 5, x.shape
        assert idx.ndim == 2 and idx.shape[1] == 3, idx.shape
        assert len(idx_max) == 3
        assert len(scale) == 3
        assert len(size) == 3
        assert len(padding) == 3

        N, T_in, H_in, W_in, C = x.shape
        T_out, H_out, W_out = tuple(size[d] // idx_max[d] + 2 * padding[d] for d in range(3))
        assert (T_out - T_in * scale[0]) % 2 == (H_out - H_in * scale[1]) % 2 == (W_out - W_in * scale[2]) % 2 == 0, \
            'Under this configuration, padding is not symmetric and can cause problems!'

        x = self.norm(x, params)

        if self.type in ['trilinear', 'nearest']:
            x = self.layer(x, idx, idx_max, size, scale, padding, patch_mode)
            assert_shape(x, (N, T_out, H_out, W_out, C))
        else:
            raise NotImplementedError

        x = self.act(x)

        return x


class Decoder(FuncModule):
    """
    HiNeRV Decoder.
    """
    def __init__(self, logger,
                 in_channels=128, out_channels=3,
                 channels=[512, 256, 128, 64], size: tuple[int, int, int]=(600, 1080, 1920),
                 depths: list=[3, 3, 3, 3], exps: list=[4, 4, 4, 1], kernels: list=[3, 3 ,3 ,3],
                 scales: list=[[1, 5, 5], [1, 3, 3], [1, 2, 2], [1, 2, 2]],
                 paddings: list=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                 use_hi_encoding: list=[True, True, True, True], num_heads: int=1, keep_paddings: bool=False,
                 upsample_cfg: dict={'type': 'trilinear', 'norm': 'layernorm', 'act': 'none', 'config': 'matmul-th-w'},
                 hi_enc_cfg: dict={'grid_size': [-1, 16],  'grid_level': 3, 'grid_level_scale': [2., 0.5],
                                   'grid_depth_scale': [1., 0.5]},
                 block_cfg: dict={'type': 'mlp', 'norm': 'layernorm', 'act': 'none', 'bias': True,
                                  'layerscale': 0.},
                 head_cfg: dict={'type': 'linear_head', 'norm': 'none', 'act': 'sigmoid', 'bias': True,
                                 'layerscale': 0.}):
        super().__init__()
        assert isinstance(depths, (tuple, list))
        assert isinstance(exps, (tuple, list))
        assert isinstance(scales, (tuple, list)) and all([isinstance(s, (tuple, list)) for s in scales])
        assert len(channels) == len(depths) == len(exps) == len(kernels) == len(scales) == \
               len(paddings) == len(use_hi_encoding)
        assert len(channels) == len(block_cfg['type']) == len(block_cfg['norm']) == len(block_cfg['act'])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.in_sizes = list(reversed(compute_multiscale_sizes(size, reversed(scales), True)))[:-1]
        self.out_sizes = list(reversed(compute_multiscale_sizes(size, reversed(scales), True)))[1:]
        self.depths = depths
        self.exps = exps
        self.kernels = kernels
        self.scales = [tuple(scales[i]) for i in range(len(scales))]
        self.paddings = [tuple(paddings[i]) for i in range(len(paddings))]
        self.use_hi_encoding = use_hi_encoding
        self.num_heads = num_heads
        self.keep_paddings = keep_paddings

        logger.info(f'Decoder:')

        self.upsamplers = nn.ModuleList()
        self.hi_encodings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()

        for i, scale in enumerate(self.scales):
            assert depths[i] > 0, 'Decoder does not support depth = 0'
            T1, H1, W1 = self.in_sizes[i]
            T2, H2, W2 = self.out_sizes[i]
            if i == 0:
                C1 = self.in_channels
                C2 = self.channels[0]
            else:
                C1 = C2
                C2 = self.channels[i]

            logger.info(f'     Stage {i + 1}:')
            logger.info(f'               T1 - {T1}  H1 - {H1}  W1 - {W1}  C1 - {C1}')
            logger.info(f'               T2 - {T2}  H2 - {H2}  W2 - {W2}  C2 - {C2}')
            logger.info(f'               Depth - {self.depths[i]}  Exp - {self.exps[i]}  Kernels - {self.kernels[i]}')
            logger.info(f'               Scale - {self.scales[i]}  Padding - {self.paddings[i]}')

            # Upsampler
            if scale != (1, 1, 1):
                _upsample_cfg = cfg_override(upsample_cfg,  type=upsample_cfg['type'][i],
                                            norm=upsample_cfg['norm'][i], act=upsample_cfg['act'][i],
                                            config=upsample_cfg['config'][i],
                                            channels=C1, scale=scale)
                self.upsamplers.append(Upsample(**_upsample_cfg))
            else:
                self.upsamplers.append(None)

            # Hierarchical Encoding
            if scale != (1, 1, 1) and self.use_hi_encoding[i]:
                _hi_enc_cfg = get_hi_encoding_cfg(C1, i, size=(T2, H2, W2, C1), scale=scale, **hi_enc_cfg)
                self.hi_encodings.append(HierarchicalEncoding(**_hi_enc_cfg))
            else:
                self.hi_encodings.append(None)

            # Block
            self.blocks.append(nn.ModuleList())

            for j in range(self.depths[i]):
                _block_cfg = cfg_override(block_cfg, type=block_cfg['type'][i],
                                        norm=block_cfg['norm'][i], act=block_cfg['act'][i],
                                        C1=C1 if j == 0 else C2, C2=C2,
                                        C_hidden=int(C2 * self.exps[i]),
                                        kernel_size=self.kernels[i])
                self.blocks[-1].append(get_func_block(**_block_cfg))

            # Head
            if i >= len(self.scales) - self.num_heads:
                assert head_cfg['type'].startswith('linear'), \
                    'Decoder head only supports linear stem. Otherwise the padding computation will be wrong.'
                _head_cfg = cfg_override(head_cfg, C1=C2, C2=self.out_channels,
                                            C_hidden=C2, kernel_size=1)
                self.heads.append(get_func_block(**_head_cfg))
            else:
                self.heads.append(None)

    def _get_params(self):
        return {}

    def forward(self, x: torch.Tensor, params: dict, idx: torch.IntTensor, idx_max: tuple[int, int, int], 
                patch_mode: bool=True):
        assert idx.ndim == 2 and idx.shape[1] == 3
        assert len(idx_max) == 3

        outputs = []

        for i in range(len(self.scales)):
            # Compute patch sizes
            scale = self.scales[i]
            padding = self.paddings[i] if patch_mode else (0, 0, 0)
            v_size = self.out_sizes[i]
            p_size = tuple(int(v_size[d] // idx_max[d]) for d in range(3))
            p_size_padded = tuple(p_size[d] + 2 * padding[d] for d in range(3))

            # Compute mask
            if patch_mode:
                _, px_mask = compute_pixel_idx_3d(idx, idx_max, v_size, padding, clipped=False, return_mask=True)
                px_mask_3d = px_mask[0][:, :, None, None, None] \
                                * px_mask[1][:, None, :, None, None] \
                                * px_mask[2][:, None, None, :, None]
            else:
                px_mask_3d = None

            # Upsample
            if self.upsamplers[i] is not None:
                x = self.upsamplers[i](x, params, idx=idx, idx_max=idx_max,
                                       size=v_size, scale=scale, padding=padding,
                                       patch_mode=patch_mode, mask=px_mask_3d)

            # Encoding
            if self.hi_encodings[i] is not None:
                x = self.hi_encodings[i](x, params, idx=idx, idx_max=idx_max,
                                         size=v_size, scale=scale, padding=padding)

            # Run block
            x = run_modules(self.blocks[i], x, params, px_mask_3d)
            assert_shape(x, (idx.shape[0],) + p_size_padded + (x.shape[-1],))

            # Output
            if i >= len(self.scales) - self.num_heads:
                y = self.heads[i](x, params, px_mask_3d) if self.heads[i] is not None else x
                y = crop_tensor_nthwc(y, p_size_padded if self.keep_paddings else p_size)
                assert_shape(y, (idx.shape[0],) + (p_size_padded if self.keep_paddings else p_size) + (y.shape[-1],))
                outputs.append(y)
            else:
                outputs.append(None)

        return outputs[-1] if self.num_heads == 1 else outputs


class HiNeRV(FuncModule):
    """
    HiNeRV
    """
    def __init__(self, logger, grid_cfg, decoder_cfg, eval_patch_size):
        super().__init__()
        self.base_encoding = create_base_encoding(logger, **grid_cfg)
        self.decoder = Decoder(logger, **decoder_cfg)
        self.eval_patch_size = eval_patch_size
        self.min_patch_size = tuple(np.prod(np.array(self.decoder.scales), axis=0).tolist())
        assert self.eval_patch_size is None or \
            all((self.eval_patch_size[d] % self.min_patch_size[d] == 0) for d in range(3)), \
            'evaluation patch size must be divisible by the minimum patch size.'
        assert self.base_encoding.out_size == self.decoder.in_sizes[0], \
            'base encoding output size must match the input size of the decoder.'
        assert self.base_encoding.out_channels == self.decoder.in_channels, \
            'base encoding output channels must match the input channels of the decoder.'

        # Initialize all functinoal modules
        self.init_func_modules()

    def _get_params(self):
        return {}

    def forward(self, inputs: dict, params: dict):
        assert all((inputs['patch_size'][d] % self.min_patch_size[d] == 0) for d in range(3)), \
              'input patch_size must be divisible by the minimum patch size.'

        # Config
        if not self.training and self.eval_patch_size is not None :
            # Force using patch mode for evaluation
            idx_max = tuple(inputs['video_size'][d] // self.eval_patch_size[d] for d in range(3))
            idx = vidx_to_pidx(inputs['idx'], inputs['idx_max'], idx_max)
            patch_mode = True
        else:
            # Auto choose frame/patch mode
            idx_max = inputs['idx_max']
            idx = inputs['idx']
            patch_mode = self.training

        # Compute output
        x = self.base_encoding(params, idx, idx_max, patch_mode=patch_mode)
        output = self.decoder(x, params, idx, idx_max, patch_mode=patch_mode)

        # Reshape output
        if not self.training and self.eval_patch_size is not None :
            output = patch_to_video(output, inputs['patch_size'])
        output = output.permute(0, 4, 1, 2, 3).contiguous()

        return output


def set_default(x, y):
    if isinstance(x, list):
        assert len(x) == len(y)
        for i in range(len(x)):
            if x[i] == -1:
                x[i] = y[i]
        return x
    else:
        return x if x != -1 else y


def expand_list(x, n):
    assert isinstance(x, list) and len(x) in [1, n]
    return x * n if len(x) == 1 else x


def _get_grid_cfg(logger, input, cfg):
    grid_cfg = {}
    grid_cfg['grid_type'] = cfg.base_encoding.base_grid_type
    grid_cfg['out_size'] = tuple(input['video_size'][d] // \
                           math.prod(cfg.decoder.scales_t if d == 0 else cfg.decoder.scales_hw) for d in range(3))
    grid_cfg['out_channels'] = cfg.decoder.channels[0]
    grid_cfg['paddings'] = (0, 0, 0)
    grid_cfg['grid_size'] = set_default(cfg.base_encoding.base_grid_size, (grid_cfg['out_size'] + (8,)))
    grid_cfg['grid_level'] = cfg.base_encoding.base_grid_level
    grid_cfg['grid_level_scale'] = cfg.base_encoding.base_grid_level_scale
    grid_cfg['kernel_size'] = cfg.base_encoding.base_kernel
    return grid_cfg


def _get_decoder_cfg(logger, input, cfg):
    n_blocks = len(cfg.decoder.channels)
    decoder_cfg = {}
    decoder_cfg['size'] = input['video_size']
    decoder_cfg['in_channels'] = cfg.decoder.channels[0]
    decoder_cfg['out_channels'] = input['channels']
    decoder_cfg['channels'] = cfg.decoder.channels
    decoder_cfg['depths'] = cfg.decoder.depths
    decoder_cfg['exps'] = cfg.decoder.exps
    decoder_cfg['kernels'] = cfg.decoder.kernels
    decoder_cfg['scales'] = [(cfg.decoder.scales_t[i], cfg.decoder.scales_hw[i], cfg.decoder.scales_hw[i]) \
                             for i in range(len(cfg.decoder.scales_t))]
    decoder_cfg['paddings'] = cfg.decoder.paddings
    decoder_cfg['use_hi_encoding'] = expand_list(cfg.decoder.use_hi_encoding, n_blocks)

    # Blocks/Heads
    for blk_type in ['block', 'head']:
        decoder_cfg[f'{blk_type}_cfg'] = {}
        if blk_type == 'block':
            decoder_cfg[f'{blk_type}_cfg']['type'] = expand_list(getattr(cfg.decoder, f'{blk_type}_type'), n_blocks)
            decoder_cfg[f'{blk_type}_cfg']['norm'] = expand_list(getattr(cfg.decoder, f'{blk_type}_norm'), n_blocks)
            decoder_cfg[f'{blk_type}_cfg']['act'] = expand_list(getattr(cfg.decoder, f'{blk_type}_act'), n_blocks)
        else:
            decoder_cfg[f'{blk_type}_cfg']['type'] = getattr(cfg.decoder, f'{blk_type}_type')
            decoder_cfg[f'{blk_type}_cfg']['norm'] = getattr(cfg.decoder, f'{blk_type}_norm')
            decoder_cfg[f'{blk_type}_cfg']['act'] = getattr(cfg.decoder, f'{blk_type}_act')
        decoder_cfg[f'{blk_type}_cfg']['layerscale'] = getattr(cfg.decoder, f'{blk_type}_layerscale')
        decoder_cfg[f'{blk_type}_cfg']['bias'] = getattr(cfg.decoder, f'{blk_type}_bias')

    # Upsampler
    decoder_cfg['upsample_cfg'] = {}
    decoder_cfg['upsample_cfg']['type'] = expand_list(getattr(cfg.decoder, 'upsample_type'), n_blocks)
    decoder_cfg['upsample_cfg']['norm'] = expand_list(getattr(cfg.decoder, 'upsample_norm'), n_blocks)
    decoder_cfg['upsample_cfg']['act'] = expand_list(getattr(cfg.decoder, 'upsample_act'), n_blocks)
    decoder_cfg['upsample_cfg']['config'] = expand_list(getattr(cfg.decoder, 'upsample_config'), n_blocks)

    # Hierarchical Encoding
    decoder_cfg['hi_enc_cfg'] = {}
    decoder_cfg['hi_enc_cfg']['local_grid_size'] = getattr(cfg.decoder, 'local_grid_size')
    decoder_cfg['hi_enc_cfg']['local_grid_level'] = getattr(cfg.decoder, 'local_grid_level')
    decoder_cfg['hi_enc_cfg']['local_grid_level_scale'] = getattr(cfg.decoder, 'local_grid_level_scale')
    decoder_cfg['hi_enc_cfg']['local_grid_depth_scale'] = getattr(cfg.decoder, 'local_grid_depth_scale')

    return decoder_cfg


def _get_model_cfg(logger, input, cfg):
    # This is not a good way to compute the required paddings and only works for most one convolution layer per stage.
    grid_cfg = _get_grid_cfg(logger, input, cfg)
    decoder_cfg = _get_decoder_cfg(logger, input, cfg)

    # Compute padding
    if all(cfg.decoder.paddings[d] == -1 for d in range(3)):
        scales = decoder_cfg['scales']
        kernels = [(0, k, k) for k in decoder_cfg['kernels']]
        depths = decoder_cfg['depths']
        paddings = compute_paddings_v2(scales=scales, kernels=kernels,
                                       depths=depths, resize_methods='trilinear')
        if cfg.base_encoding.base_grid_type == 'v1':
            grid_cfg['paddings'] = tuple(paddings[0][d] + \
                                         (0 if d == 0 else grid_cfg['kernel_size'][d] // 2) for d in range(3))
        elif cfg.base_encoding.base_grid_type == 'v2':
            grid_cfg['paddings'] = paddings[0]
        else:
            raise NotImplementedError
        decoder_cfg['paddings'] = paddings[1:]
    elif all(cfg.decoder.paddings[d] == 0 for d in range(3)):
        grid_cfg['paddings'] = [tuple(cfg.decoder.paddings)] * cfg.decoder.base_num_grids
        decoder_cfg['paddings'] = [tuple(cfg.decoder.paddings)] * len(decoder_cfg['channels'])
    else:
        raise NotImplementedError

    return grid_cfg, decoder_cfg


def build_model(args, logger, input):
    cfg = HiNeRVConfig(**args)
    encoding_cfg, decoder_cfg = _get_model_cfg(logger, input, cfg)
    return HiNeRV(logger, encoding_cfg, decoder_cfg, cfg.eval_patch_size)