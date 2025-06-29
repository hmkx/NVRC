from .utils import *


def _add_kumaraswamy_noise(inputs, kum):
    a = kum
    b = (1. / a) * (2 ** a * (a - 1) + 1)
    return inputs + ((1 - torch.rand_like(inputs) ** (1 / b)) ** (1 / a) - 0.5)


def _soft_round(x, temperature):
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / temperature)
    r = torch.tanh((x - m) / temperature) / z
    return m + r


def _soft_round_inverse(x, temperature):
    m = x.floor() + 0.5
    z = 2 * torch.tanh(0.5 / temperature)
    r = torch.atanh((x - m) * z) * temperature
    return m + r


def _soft_round_conditional_mean(x, temperature):
    return _soft_round_inverse(x - 0.5, temperature) + 0.5


class QuantizerBase(FuncModule):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, params):
        raise NotImplementedError


class NoneQuantizer(QuantizerBase):
    def forward(self, inputs, params):
        if self.training:
            return inputs
        else:
            return inputs.round()


class UniformNoiseQuantizer(QuantizerBase):
    def forward(self, inputs, params):
        if self.training:
            return inputs + (torch.rand_like(inputs) - 0.5)
        else:
            return inputs.round()


class STEQuantizer(QuantizerBase):
    def forward(self, inputs, params):
        if self.training:
            return compressai.ops.quantize_ste(inputs)
        else:
            return inputs.round()


class SoftRoundingQuantizer(QuantizerBase):
    """
    See https://github.com/google-deepmind/c3_neural_compression/blob/main/model/latents.py
    """
    def __init__(self, use_noise):
        super().__init__()
        self.use_noise = use_noise
        self.register_buffer('temp', torch.tensor(1., dtype=torch.float32), persistent=False)
        self.register_buffer('kum', torch.tensor(2., dtype=torch.float32), persistent=False)

    def extra_repr(self):
        s = 'use_noise={use_noise}'
        return s.format(**self.__dict__)

    def set_params(self, temp, kum):
        self.temp.data.fill_(temp)
        self.kum.data.fill_(kum)

    def forward(self, inputs, params):
        if self.training:
            return _soft_round_conditional_mean(
                _add_kumaraswamy_noise(_soft_round(inputs, self.temp), self.kum), self.temp
            )
        else:
            return inputs.round()


class QuantNoiseQuantizer(QuantizerBase):
    def __init__(self, use_ste):
        super().__init__()
        self.use_ste = use_ste
        self.register_buffer('noise_ratio', torch.tensor(0., dtype=torch.float32), persistent=False)

    def extra_repr(self):
        s = 'use_ste={use_ste}'
        return s.format(**self.__dict__)

    def set_params(self, noise_ratio):
        self.noise_ratio.data.fill_(noise_ratio)

    def forward(self, inputs, params):
        if self.training:
            mask = (torch.rand_like(inputs) > self.noise_ratio).to(inputs.dtype)
            if self.use_ste:
                return mask * inputs + (1. - mask) * compressai.ops.quantize_ste(inputs)
            else:
                return mask * inputs + (1. - mask) * inputs.round()
        else:
            return inputs.round()


def get_quantizer(quantizer, **kwargs):
    if quantizer == 'none':
        return NoneQuantizer()
    elif quantizer == 'uniform-noise':
        return UniformNoiseQuantizer()
    elif quantizer == 'ste':
        return STEQuantizer()
    elif quantizer == 'soft-round-noise':
        return SoftRoundingQuantizer(use_noise=True)
    elif quantizer == 'quant-noise':
        return QuantNoiseQuantizer(use_ste=False)
    elif quantizer == 'quant-noise-ste':
        return QuantNoiseQuantizer(use_ste=True)
    else:
        raise ValueError


def update_quantizer(quantizer, **kwargs):
    if type(quantizer) == NoneQuantizer:
        pass
    elif type(quantizer) == UniformNoiseQuantizer:
        pass
    elif type(quantizer) == STEQuantizer:
        pass
    elif type(quantizer) == SoftRoundingQuantizer:
        quantizer.set_params(kwargs['soft_round_temp'], kwargs['kum_noise'])
    elif type(quantizer) == QuantNoiseQuantizer:
        quantizer.set_params(kwargs['quant_noise'])
    else:
        pass