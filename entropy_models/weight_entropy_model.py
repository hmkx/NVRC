"""
Weight entropy model classes
"""
from .utils import *
from .entropy_model_base import EntropyModel


class WeightEntropyModel(EntropyModel):
    def __init__(self, logger, name, inputs, config):
        super(EntropyModel, self).__init__(logger, name, inputs, config)
        self.learned_quant = config.learned_quant
        self.learned_em = config.learned_em

        self.blk_size = config.blk_size
        self.tensor_keys = self._build_keys(inputs)
        self.tensor_cfgs = self._build_cfgs(inputs)
        self.forward_order = self._build_forward_order(inputs)
        self._check_implementation(inputs)
        self._build_masks()

    def extra_repr(self):
        s = super().extra_repr() + ', blk_size={blk_size}'
        return s.format(**self.__dict__)

    def _build_cfgs(self, inputs):
        assert len(self.blk_size) == 2 and all(s > 0 for s in self.blk_size)

        tensor_cfgs = {}
        offset = 0

        with torch.no_grad():
            for tensor_k, tensor in self._sort_tensors(inputs):
                tensor = tensor.unsqueeze(0)
                reshape = (-1, tensor.shape[1], math.prod(tensor.shape[2:]))
                padding = tuple((p - s % p) % p for s, p in zip(reshape[1:], self.blk_size))
                reshape_pad = (reshape[0], reshape[1] + padding[0], reshape[2] + padding[1])
                num_blks = tuple(s // p for s, p in zip(reshape_pad[1:], self.blk_size))
                length = math.prod(num_blks)
                tensor_cfgs[tensor_k] = {}
                tensor_cfgs[tensor_k]['shape'] = (-1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['mask_shape'] = (1, *tensor.shape[1:])
                tensor_cfgs[tensor_k]['reshape'] = reshape
                tensor_cfgs[tensor_k]['reshape_pad'] = reshape_pad
                tensor_cfgs[tensor_k]['reshape_blk'] = (-1, length, math.prod(self.blk_size))
                tensor_cfgs[tensor_k]['padding'] = (0, padding[1], 0, padding[0])
                tensor_cfgs[tensor_k]['unpadding'] = (0, -padding[1], 0, -padding[0])
                tensor_cfgs[tensor_k]['num_blks'] = num_blks
                tensor_cfgs[tensor_k]['offset'] = offset
                tensor_cfgs[tensor_k]['length'] = length
                tensor_cfgs[tensor_k]['meta_shape'] = (length, 1)
                offset += length

        return tensor_cfgs

    def _reshape_tensor(self, tensor_k, tensor):
        tensor_cfg = self.tensor_cfgs[tensor_k]
        tensor = tensor.unsqueeze(0)
        tensor = F.pad(tensor.view(tensor_cfg['reshape']), tensor_cfg['padding'])
        tensor = tensor.view(tensor.shape[0], tensor_cfg['num_blks'][0], self.blk_size[0],
                             tensor_cfg['num_blks'][1], self.blk_size[1]) \
                       .permute(0, 1, 3, 2, 4).contiguous().view(tensor_cfg['reshape_blk'])
        return tensor

    def _restore_tensor(self, tensor_k, tensor):
        tensor_cfg = self.tensor_cfgs[tensor_k]
        tensor = tensor.view(tensor.shape[0], *tensor_cfg['num_blks'], *self.blk_size) \
                       .permute(0, 1, 3, 2, 4).contiguous().view(tensor_cfg['reshape_pad'])
        tensor = F.pad(tensor, tensor_cfg['unpadding']).view(tensor_cfg['shape'])
        tensor = tensor.squeeze(0)
        return tensor.contiguous()

    def _preprocess(self, inputs):
        outputs = []
        for tensor_k, tensor in self._sort_tensors(inputs):
            outputs.append(self._reshape_tensor(tensor_k, tensor))
        return {'all': torch.cat(outputs, dim=1)}

    def _postprocess(self, inputs):
        outputs = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            outputs[tensor_k] = self._restore_tensor(
                tensor_k, inputs['all'][:, tensor_cfg['offset']:tensor_cfg['offset'] + tensor_cfg['length']]
            )
        return outputs

    def _init_meta(self, k, x, params):
        if self.learned_em:
            meta = {}
        else:
            meta = {
                'means': masked_mean(x, getattr(self, f'{k}_mask'), dim=2, keepdim=True),
                'scales': masked_std(x, getattr(self, f'{k}_mask'), dim=2, keepdim=True) \
                          if x.numel() > 1 else torch.zeros_like(x)
            }
        return meta

    def _extract_meta(self, k, i, meta):
        return meta

    def _extract_code(self, k, i, ctx, ctx_mask, x, params):
        return x, getattr(self, f'{k}_mask').expand_as(x)

    def _entropy_param(self, k, i, ctx_i, ctx_mask_i, meta_i, params):
        if self.learned_em:
            means = []
            scales = []
            for tensor_k, cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_means = params[tensor_k + f'_means']
                tensor_scales = params[tensor_k + f'_log_scales'].exp()
                tensor_means = tensor_means.unsqueeze(0)
                tensor_scales = tensor_scales.unsqueeze(0)
                means.append(tensor_means.expand(cfg['reshape_blk']))
                scales.append(tensor_scales.expand(cfg['reshape_blk']))
            means = getattr(self, f'{k}_mask') * torch.cat(means, dim=1)
            scales = getattr(self, f'{k}_mask') * torch.cat(scales, dim=1)
        else:
            means = getattr(self, f'{k}_mask') * meta_i['means'].float()
            scales = getattr(self, f'{k}_mask') * meta_i['scales'].float()

        return means, scales

    def _get_log_step(self, k, i, params):
        if self.learned_quant:
            log_steps = []
            for tensor_k, cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_log_step = params[tensor_k + '_log_step']
                tensor_log_step = tensor_log_step.unsqueeze(0)
                log_steps.append(tensor_log_step.expand(cfg['reshape_blk']))
            return torch.concat(log_steps, dim=1)
        else:
            return -4.


class _AxisEntropyModelBase(EntropyModel):
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
                tensor_cfgs[tensor_k]['meta_shape'] = self._get_meta_shapes(tensor_k, tensor) # [(shape, dim), ...]
                offset += length

        return tensor_cfgs

    def _get_meta_shapes(self, k, x):
        raise NotImplementedError

    def _get_params(self):
        params = {}
        for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
            for i, (meta_shape, _) in enumerate(tensor_cfg['meta_shape']):
                # Log step
                if self.learned_quant:
                    params[tensor_k + f'_log_step_{i}'] = init_log_step(meta_shape)
                # EM parameters
                if self.learned_em:
                    params[tensor_k + f'_means_{i}'] = init_means(meta_shape)
                    params[tensor_k + f'_log_scales_{i}'] = init_log_scales(meta_shape)
        return params

    def _set_params(self, inputs, init_bit_width, params):
        with torch.no_grad():
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                # Log step
                if self.learned_quant and init_bit_width is not None:
                    x = self._reshape_tensor(tensor_k, inputs[tensor_k])
                    x = x.unsqueeze(0)
                    for i, (_, dim) in enumerate(tensor_cfg['meta_shape']):
                        # Get log steps
                        log_step = params[tensor_k + f'_log_step_{i}']
                        log_step = log_step.unsqueeze(0)

                        # Set width
                        set_width(x.view(-1, tensor_cfg['shape'][1], math.prod(tensor_cfg['shape'][2:])),
                                  log_step, init_bit_width, dim)

    def _init_meta(self, k, x, params):
        if self.learned_em:
            meta = {}
        else:
            meta = {}
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor = x[:, :, tensor_cfg['offset']:tensor_cfg['offset'] + tensor_cfg['length']]
                tensor = tensor.view(-1, tensor_cfg['shape'][1], math.prod(tensor_cfg['shape'][2:]))
                for i, (_, dim) in enumerate(tensor_cfg['meta_shape']):
                    means_i = tensor.mean(dim=dim, keepdim=True)
                    scales_i = tensor.std(dim=dim, keepdim=True) if tensor.numel() > 1 else torch.zeros_like(tensor)
                    meta[tensor_k + f'_means_{i}'] = means_i.half()
                    meta[tensor_k + f'_scales_{i}'] = scales_i.half()
        return meta

    def _extract_meta(self, k, i, meta):
        return meta

    def _entropy_param(self, k, d, ctx_i, ctx_mask_i, meta_i, params):
        means = []
        scales = []

        for tensor_k, cfg in self._sort_tensors(self.tensor_cfgs):
            tensor_means = 0.
            tensor_scales = 1.

            for d, (_, _) in reversed(list(enumerate(cfg['meta_shape']))):
                if self.learned_em:
                    tensor_means_d = params[tensor_k + f'_means_{d}']
                    tensor_scales_d = params[tensor_k + f'_log_scales_{d}'].exp()
                    tensor_means_d = tensor_means_d.unsqueeze(0)
                    tensor_scales_d = tensor_scales_d.unsqueeze(0)
                else:
                    tensor_means_d = meta_i[tensor_k + f'_means_{d}'].float()
                    tensor_scales_d = meta_i[tensor_k + f'_scales_{d}'].float()
                tensor_means_d = tensor_means_d.expand(-1, cfg['shape'][1], math.prod(cfg['shape'][2:])) \
                                               .contiguous().view(cfg['reshape'])
                tensor_scales_d = tensor_scales_d.expand(-1, cfg['shape'][1], math.prod(cfg['shape'][2:])) \
                                                 .contiguous().view(cfg['reshape'])
                tensor_means = tensor_means * tensor_scales_d + tensor_means_d
                tensor_scales = tensor_scales * tensor_scales_d

            means.append(tensor_means)
            scales.append(tensor_scales)

        means = torch.concat(means, dim=2)
        scales= torch.concat(scales, dim=2)

        return means, scales

    def _get_log_step(self, k, i, params):
        if self.learned_quant:
            log_steps = []
            for tensor_k, tensor_cfg in self._sort_tensors(self.tensor_cfgs):
                tensor_log_step = 0.
                for d, (_, _) in enumerate(tensor_cfg['meta_shape']):
                    tensor_log_step_d = params[tensor_k + f'_log_step_{d}']
                    tensor_log_step_d = tensor_log_step_d.unsqueeze(0)
                    tensor_log_step += tensor_log_step_d.expand(-1, tensor_cfg['shape'][1],
                                                                math.prod(tensor_cfg['shape'][2:])) \
                                                        .contiguous().view(tensor_cfg['reshape'])
                log_steps.append(tensor_log_step / len(tensor_cfg['meta_shape']))
            return torch.concat(log_steps, dim=2)
        else:
            return -4.


class SingleAxisEntropyModel(_AxisEntropyModelBase):
    def _get_meta_shapes(self, tensor_k, tensor):
        meta_shapes = []
        if tensor.ndim >= 3:
            if tensor.shape[1] >= 2 ** 7 and tensor.shape[1] >= math.prod(tensor.shape[2:]):
                meta_shapes.append(((1, math.prod(tensor.shape[2:])), 1))
            elif math.prod(tensor.shape[2:]) >= 2 ** 7 and math.prod(tensor.shape[2:]) >= tensor.shape[1]:
                meta_shapes.append(((tensor.shape[1], 1), 2))
            else:
                meta_shapes.append(((1, 1), None))
        else:
            meta_shapes.append(((1, 1), None))
        return meta_shapes


class DoubleAxisEntropyModel(_AxisEntropyModelBase):
    def _get_meta_shapes(self, tensor_k, tensor):
        meta_shapes = []
        if tensor.ndim >= 3:
            if tensor.shape[1] >= 2 ** 7:
                meta_shapes.append(((1, math.prod(tensor.shape[2:])), 1))
            if math.prod(tensor.shape[2:]) >= 2 ** 7:
                meta_shapes.append(((tensor.shape[1], 1), 2))
            if len(meta_shapes) == 0:
                meta_shapes.append(((1, 1), None))
        else:
            meta_shapes.append(((1, 1), None))
        return meta_shapes