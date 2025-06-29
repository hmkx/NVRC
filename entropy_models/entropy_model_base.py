"""
Basic entropy model classes
"""
from .utils import *
from .distributions import *
from .quantizer import *


class FullPrecisionModel(FuncModule):
    def __init__(self, logger, name, inputs):
        super().__init__()
        self.name = name

    def forward(self, x, params={}, forward_rates=True, forward_params=True):
        rates, x = forward_meta(x)
        return (sum(rates.values()) if forward_rates else None), (x if forward_params else None)

    def compress(self, bs, x, params={}):
        return compress_meta(bs, x)

    def decompress(self, bs, x, params={}):
        return decompress_meta(bs, x)


class HalfPrecisionModel(FullPrecisionModel):
    def _convert_half(self, x):
        x_half = {}
        for k, v in x.items():
            x_half[k] = v.half() if v is not None and isinstance(v, torch.Tensor) and v.is_floating_point() else v
        return x_half

    def _restore_half(self, x_half, x_ref):
        x = {}
        for k, v in x_half.items():
            x[k] = v.to(x_ref[k].dtype) if v is not None and isinstance(v, torch.Tensor) else v
        return x

    def forward(self, x, params={}, forward_rates=True, forward_params=True):
        x_half = self._convert_half(x)
        return super().forward(x_half, params, forward_rates, forward_params)

    def compress(self, bs, x, params={}):
        x_half = self._convert_half(x)
        bs, rate, x_half = super().compress(bs, x_half)
        return bs, rate, self._restore_half(x_half, x)

    def decompress(self, bs, x, params={}):
        x_half = self._convert_half(x)
        bs, rate, x_half = super().decompress(bs, x_half)
        return bs, rate, self._restore_half(x_half, x)


class _EntropyModelBase(FuncModule):
    """
    The basic entropy model class.
    """
    def __init__(self, logger, name, inputs, config):
        super().__init__()
        self.name = name
        if config.em_dist == 'gaussian':
            self.dist_model = GaussianConditional(
                scale_table=np.exp(np.linspace(np.log(0.11), np.log(256), 64)).tolist(), scale_bound=0.1
            )
        else:
            raise ValueError

        # Quantisation settings
        self.rate_quantizer = get_quantizer(config.rate_quant)
        self.dist_quantizer = get_quantizer(config.dist_quant)

        # Flag to indicate whether the model is in compression mode.
        self.compress_mode = 'none'

        # Flag to indicate whether the means are quantised. 
        # This is used to ensure the consistency between the estimation and the actual coding.
        self.use_quant_mean = config.quant_mean

    def extra_repr(self):
        s = 'use_quant_mean={use_quant_mean}'
        return s.format(**self.__dict__)

    def _check_implementation(self, inputs):
        # Test pre- and post-processing
        inputs_restored = self._postprocess(self._preprocess(inputs))
        for k in inputs.keys():
            assert (inputs[k] is None and k not in inputs_restored) or torch.allclose(inputs[k], inputs_restored[k]), \
                f'Input {k} is not correctly before/after pre- and post-processing.'

    def _set_params(self, inputs, init_bit_width, params):
        pass

    def set_params(self, inputs, init_bit_width, soft_round_temp, kum_noise, quant_noise, params):
        with torch.no_grad():
            # Update quantisers
            update_quantizer(self.rate_quantizer,
                             soft_round_temp=soft_round_temp, kum_noise=kum_noise,
                             quant_noise=quant_noise)
            update_quantizer(self.dist_quantizer, 
                             soft_round_temp=soft_round_temp, kum_noise=kum_noise,
                             quant_noise=quant_noise)
            self._set_params(inputs, init_bit_width, params)

    def _preprocess(self, inputs):
        raise NotImplementedError

    def _postprocess(self, inputs):
        raise NotImplementedError

    def _num_iters(self, k):
        raise NotImplementedError

    def _init_ctx(self, k, x, params):
        raise NotImplementedError

    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        raise NotImplementedError

    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        raise NotImplementedError

    def _init_meta(self, k, x, params):
        raise NotImplementedError

    def _extract_meta(self, k, i, meta):
        raise NotImplementedError

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        raise NotImplementedError

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        raise NotImplementedError

    def _get_log_step(self, k, i, params):
        raise NotImplementedError

    def _scale(self, k, i, x_i, params):
        return scale(x_i, self._get_log_step(k, i, params))

    def _unscale(self, k, i, x_i, params):
        return unscale(x_i, self._get_log_step(k, i, params))

    def _compute_rate(self, k, i, p_i, p_i_mask):
        return -torch.log2(p_i) * p_i_mask if p_i_mask is not None else -torch.log2(p_i)

    def _forward_rate_tensor(self, k, x, params, forward_rate=True, forward_tensor=True, reduce_rates=True):
        assert forward_rate or forward_tensor, 'At least one of forward_rates and forward_tensor should be True'
        self.compress_mode = 'none'

        # Init rate
        if reduce_rates:
            rate = 0.
        else:
            rate = {'meta': [], 'data': []}

        # Scale
        # TODO: Extract the log step & prevent re-computation for scale/unscale
        x_s = self._scale(k, None, x, params)

        if forward_rate:
            # Quantisation
            if type(self.rate_quantizer) == type(self.dist_quantizer):
                x_rsq = x_dsq = self.rate_quantizer(x_s, params)
            else:
                x_rsq = self.rate_quantizer(x_s, params)
                x_dsq = self.dist_quantizer(x_s, params)

            # Unscale
            x_dq = self._unscale(k, None, x_dsq, params)
        else:
            # Quantisation
            x_dsq = self.dist_quantizer(x_s, params)

            # Unscale
            x_dq = self._unscale(k, None, x_dsq, params)

            return None, x_dq

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_dq, params)
        assert ctx is None or is_shape_the_same(ctx, x), \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or is_shape_compatible(ctx_mask, ctx), \
            'ctx_mask should have a compatible shape with ctx.'

        # Init meta data
        meta = self._init_meta(k, x_dq, params)
        meta_rates, meta = forward_meta(meta)
        if reduce_rates:
            rate += sum(meta_rates.values())
        else:
            rate['meta'].append(sum(meta_rates.values()))

        for i in range(self._num_iters(k)):
            # Extract context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract codes to be encode
            x_rsq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_rsq, params)
            x_dsq_i, _ = self._extract_code(k, i, ctx, ctx_mask, x_dsq, params)
            assert x_i_mask is None or is_shape_compatible(x_i_mask, x_rsq_i), \
                'x_i_mask should have a compatible shape with x_rsq_i.'
            assert is_shape_the_same(x_rsq_i, x_dsq_i), \
                'x_sqn_i should have the same shape as x_sq_i.'

            # Encoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert is_shape_compatible(means, x_rsq_i), \
                    'means should have a compatible shape with x_sq_i.'
                assert is_shape_compatible(scales, x_rsq_i), \
                    'scales should have a compatible shape with x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params)
                scales_s = self._scale(k, i, scales, params)

                if self.use_quant_mean:
                    means_s = self.dist_quantizer(means_s, params)

                # Likelihood
                p_i = self.dist_model(x_rsq_i, means=means_s, scales=scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_qn_i = self._unscale(k, i, x_dsq_i, params)

            # Accumulate rate
            rate_i = self._compute_rate(k, i, p_i, x_i_mask)
            if reduce_rates:
                rate += rate_i.sum()
            else:
                rate['data'].append(rate_i)

            # Update context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_qn_i, params)
            assert ctx is None or is_shape_the_same(ctx, x), \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or is_shape_compatible(ctx_mask, ctx), \
                'ctx_mask should have a compatible shape with ctx.'

        return rate, (x_dq if forward_tensor else None)

    def _sort_forward(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, params, forward_rates=True, forward_params=True):
        assert forward_rates or forward_params, 'At least one of forward_rates and forward_params should be True'
        rates = {}
        outputs = {}
        inputs = self._preprocess(inputs)

        for k, x in self._sort_forward(inputs):
            x_rate, x_q = self._forward_rate_tensor(
                k, x, params, forward_rate=forward_rates, forward_tensor=forward_params
            )
            rates[k] = x_rate
            outputs[k] = x_q

        return (sum(rates.values()) if forward_rates else None), \
               (self._postprocess(outputs) if forward_params else None) 

    def _compress_tensor(self, bs, k, x, params):
        assert not self.training, 'compression should be done in eval mode.'
        self.compress_mode = 'encoding'
        self.dist_model.update()

        # Init bitstreams
        meta_bs = b''
        data_bs = b''

        # Scale
        x_s = self._scale(k, None, x, params)

        # Quantisation
        x_sq = x_s.round()

        # Unscale
        x_q = self._unscale(k, None, x_sq, params)

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_q, params)
        assert ctx is None or ctx.shape == x.shape, \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or ctx_mask.shape == x.shape, \
            'ctx_mask should have the same shape as x.'

        # Init meta data
        meta = self._init_meta(k, x_q, params)
        meta_bs, _, meta = compress_meta(meta_bs, meta)

        # Init encoder buffer
        self.dist_model.encode_buffer_init()

        for i in range(self._num_iters(k)):
            # Extract condition and context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract codes to be encode
            x_sq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_sq, params)
            assert x_i_mask is None or x_sq_i.shape == x_i_mask.shape, \
                'x_i_mask should have the same shape as x_sq_i.'

            # Encoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert means.shape == x_sq_i.shape, \
                    'means should have the same shape as x_sq_i.'
                assert scales.shape == x_sq_i.shape, \
                    'scales should have the same shape as x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params).round()
                scales_s = self._scale(k, i, scales, params)
                if x_i_mask is not None:
                    means_s = means_s * x_i_mask
                    scales_s = scales_s * x_i_mask

                # Accumulate outputs to the buffer
                self.dist_model.encode_buffer_append(x_sq_i, means_s, scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_q_i = self._unscale(k, i, x_sq_i, params)

            # Refill context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_q_i, params)
            assert ctx is None or x.shape == ctx.shape, \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or x.shape == ctx_mask.shape, \
                'ctx_mask should have the same shape as x.'

        # Encoding
        data_bs += self.dist_model.encode_with_buffer()

        # Combine all bitstreams
        header = {'meta_length': torch.tensor(len(meta_bs), dtype=torch.int64), 
                  'data_length': torch.tensor(len(data_bs), dtype=torch.int64)}
        bs, header_rates, header = compress_meta(bs, header)
        rate = sum(header_rates.values())

        bs += meta_bs
        rate += len(meta_bs) * 8

        bs += data_bs
        rate += len(data_bs) * 8

        return bs, rate, ctx

    def _decompress_tensor(self, bs, k, x, params):
        assert not self.training, 'compression should be done in eval mode.'
        self.compress_mode = 'decoding'
        self.dist_model.update()

        # Init bitstreams
        meta_bs = b''
        data_bs = b''

        # Split bitstream
        header = {'meta_length': torch.tensor(0, dtype=torch.int64), 
                  'data_length': torch.tensor(0, dtype=torch.int64)}
        bs, header_rates, header = decompress_meta(bs, header)
        rate = sum(header_rates.values())

        meta_bs, bs = bs[:header['meta_length']], bs[header['meta_length']:] 
        rate += len(meta_bs) * 8

        data_bs, bs = bs[:header['data_length']], bs[header['data_length']:] 
        rate += len(data_bs) * 8

        # Scale
        x_s = self._scale(k, None, x, params)

        # Quantisation
        x_sq = x_s.round()

        # Unscale
        x_q = self._unscale(k, None, x_sq, params)

        # Init contexts
        ctx, ctx_mask = self._init_ctx(k, x_q, params)
        assert ctx is None or x.shape == ctx.shape, \
            'ctx should have the same shape as x.'
        assert ctx_mask is None or x.shape == ctx_mask.shape, \
            'ctx_mask should have the same shape as x.'

        # Init meta data
        meta = self._init_meta(k, x_q, params)
        meta_bs, _, meta = decompress_meta(meta_bs, meta)

        # Init decoder buffer
        self.dist_model.decode_buffer_init(data_bs)

        for i in range(self._num_iters(k)):
            # Extract condition and context
            ctx_i, ctx_mask_i = self._extract_ctx(k, i, ctx, ctx_mask, params)

            # Extract meta data
            meta_i = self._extract_meta(k, i, meta)

            # Extract template codes
            x_sq_i, x_i_mask = self._extract_code(k, i, ctx, ctx_mask, x_sq, params)
            assert x_i_mask is None or x_sq_i.shape == x_i_mask.shape, \
                'x_i_mask should have the same shape as x_sq_i.'
 
            # Decoding step
            if type(self.dist_model) in [GaussianConditional]:
                # Entropy parameters
                means, scales = self._entropy_param(k, i, ctx_i, ctx_mask_i, meta_i, params)
                assert means.shape == x_sq_i.shape, \
                    'means should have the same shape as x_sq_i.'
                assert scales.shape == x_sq_i.shape, \
                    'scales should have the same shape as x_sq_i.'

                # Scaling means and scales
                means_s = self._scale(k, i, means, params).round()
                scales_s = self._scale(k, i, scales, params)
                if x_i_mask is not None:
                    means_s = means_s * x_i_mask
                    scales_s = scales_s * x_i_mask

                # Decoding
                x_sq_i = self.dist_model.decode_with_buffer(x_sq_i, means_s, scales_s)
            else:
                raise NotImplementedError

            # Unscaling
            x_q_i = self._unscale(k, i, x_sq_i, params)

            # Refill context
            ctx, ctx_mask = self._fill_ctx(k, i, ctx, ctx_mask, x_q_i, params)
            assert ctx is None or x.shape == ctx.shape, \
                'ctx should have the same shape as x.'
            assert ctx_mask is None or x.shape == ctx_mask.shape, \
                'ctx_mask should have the same shape as x.'

        return bs, rate, ctx

    def compress(self, bs, inputs, params):
        with torch.no_grad():
            rate = 0
            outputs = {}
            inputs = self._preprocess(inputs)

            for k, x in self._sort_forward(inputs):
                bs, x_rate, x = self._compress_tensor(bs, k, x, params)
                rate += x_rate
                outputs[k] = x

            outputs = self._postprocess(outputs)

            return bs, rate, outputs

    def decompress(self, bs, inputs, params):
        with torch.no_grad():
            rate = 0
            outputs = {}
            inputs = self._preprocess(inputs)

            for k, x in self._sort_forward(inputs):
                bs, x_rate, x = self._decompress_tensor(bs, k, x, params)
                rate += x_rate
                outputs[k] = x

            outputs = self._postprocess(outputs)

            return bs, rate, outputs


class EntropyModel(_EntropyModelBase):
    def __init__(self, logger, name, inputs, config):
        super().__init__(logger, name, inputs, config)
        self.learned_quant = config.learned_quant
        self.learned_em = config.learned_em
        self.tensor_keys = self._build_keys(inputs)
        self.tensor_cfgs = self._build_cfgs(inputs)
        self.forward_order = self._build_forward_order(inputs)
        self.length = sum([tensor_cfg['length'] for tensor_cfg in self.tensor_cfgs.values()])
        self._check_implementation(inputs)
        self._build_masks()

    def extra_repr(self):
        s = super().extra_repr() + ', learned_quant={learned_quant}, learned_em={learned_em}'
        return s.format(**self.__dict__)

    def _sort_tensors(self, inputs):
        return [(tensor_k, inputs[tensor_k]) for tensor_k in self.tensor_keys]

    def _build_keys(self, inputs):
        return sorted(tensor_k for tensor_k in inputs.keys() if inputs[tensor_k] is not None)

    def _build_cfgs(self, inputs):
        tensor_cfgs = {}
        offset = 0

        with torch.no_grad():
            for tensor_k, tensor in self._sort_tensors(inputs): 
                tensor = tensor.unsqueeze(0)
                length = math.prod(tensor.shape[1:])
                tensor_cfgs[tensor_k] = {}
                tensor_cfgs[tensor_k]['shape'] = (-1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['reshape'] = (-1, 1, math.prod(tensor.shape[1:]))
                tensor_cfgs[tensor_k]['offset'] = offset
                tensor_cfgs[tensor_k]['length'] = length
                tensor_cfgs[tensor_k]['meta_shape'] = (1, 1)
                offset += length

        return tensor_cfgs

    def _build_forward_order(self, inputs):
        return ['all']

    def _build_masks(self):
        masks = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            if tensor_k in masks and masks[tensor_k] is not None:
                masks[tensor_k] = masks[tensor_k].bool()
                continue
            masks[tensor_k] = torch.ones(tensor_cfg['shape'][1:], dtype=torch.bool)

        for k, m in self._preprocess(masks).items():
            self.register_buffer(f'{k}_mask', m, persistent=False)

    def _get_params(self):
        params = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            meta_shape = tensor_cfg['meta_shape']
            # Log step
            if self.learned_quant:
                params[f'{tensor_k}_log_step'] = init_log_step(meta_shape)
            # EM parameters
            if self.learned_em:
                params[f'{tensor_k}_means'] = init_means(meta_shape, means=0.)
                params[f'{tensor_k}_log_scales'] = init_log_scales(meta_shape, scales=1.0)
        return params

    def _set_params(self, inputs, init_bit_width, params):
        with torch.no_grad():
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                # Log step
                if self.learned_quant and init_bit_width is not None:
                    # Get log steps
                    log_step = params[f'{tensor_k}_log_step']
                    log_step = log_step.unsqueeze(0)

                    # Set width
                    set_width(self._reshape_tensor(tensor_k, inputs[tensor_k]), log_step, init_bit_width)

    def _reshape_tensor(self, tensor_k, tensor):
        tensor = tensor.unsqueeze(0)
        tensor = tensor.view(self.tensor_cfgs[tensor_k]['reshape'])
        return tensor

    def _restore_tensor(self, tensor_k, tensor):
        tensor = tensor.view(self.tensor_cfgs[tensor_k]['shape'])
        tensor = tensor.squeeze(0)
        return tensor.contiguous()

    def _preprocess(self, inputs):
        outputs = []
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs.append(self._reshape_tensor(tensor_k, tensor))
        return {'all': torch.cat(outputs, dim=2)}

    def _postprocess(self, inputs):
        outputs = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            outputs[tensor_k] = self._restore_tensor(
                tensor_k, inputs['all'][:, :, tensor_cfg['offset']:tensor_cfg['offset'] + tensor_cfg['length']]
            )
        return outputs

    def _num_iters(self, k):
        return 1

    def _init_ctx(self, k, x, params):
        return None, None

    def _extract_ctx(self, k, i, ctx, ctx_mask, params):
        return None, None

    def _fill_ctx(self, k, i, ctx, ctx_mask, x_i, params):
        return x_i, ctx_mask

    def _init_meta(self, k, x, params):
        if self.learned_em:
            meta = {}
        else:
            meta = {
                'means': x.mean(dim=2, keepdims=True).half(),
                'scales': x.std(dim=2, keepdims=True).half() if x.shape[2] > 1 else torch.zeros_like(x).half()
            }
        return meta

    def _extract_meta(self, k, i, meta):
        return meta

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        return x, getattr(self, f'{k}_mask')

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        if self.learned_em:
            means = []
            scales = []

            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_means = params[f'{tensor_k}_means']
                tensor_scales = params[f'{tensor_k}_log_scales'].exp()
                tensor_means = tensor_means.unsqueeze(0)
                tensor_scales = tensor_scales.unsqueeze(0)
                means.append(tensor_means.expand(tensor_cfg['reshape']))
                scales.append(tensor_scales.expand(tensor_cfg['reshape']))

            means = torch.cat(means, dim=2)
            scales = torch.cat(scales, dim=2)
        else:
            means = meta_i['means'].float().expand(-1, -1, self.length)
            scales = meta_i['scales'].float().expand(-1, -1, self.length)
        return means, scales

    def _sort_forward(self, inputs):
        return [(k, inputs[k]) for k in self.forward_order]

    def _get_log_step(self, k, i, params):
        if self.learned_quant:
            log_steps = []
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_log_step = params[f'{tensor_k}_log_step']
                tensor_log_step = tensor_log_step.unsqueeze(0)
                log_steps.append(tensor_log_step.expand(tensor_cfg['reshape']))
            return torch.concat(log_steps, dim=2)
        else:
            return -4.