
from .utils import *


class InnerGaussianConditional(compressai.entropy_models.GaussianConditional):
    pass


class GaussianConditional(nn.Module):
    """
    - See https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/entropy_models/entropy_models.py
    """
    def __init__(self, scale_table=None, *args, **kwargs):
        nn.Module.__init__(self)
        self.inner_module = InnerGaussianConditional(scale_table, *args, **kwargs)
        self.encoder = compressai.ans.BufferedRansEncoder()
        self.decoder = compressai.ans.RansDecoder()
        self.quantized_cdf_list = None
        self.cdf_length_list = None
        self.offset_list = None
        self.symbols_list = None
        self.indexes_list = None
        self.inner_module.update()

    def update(self):
        self.inner_module.update()
        self.quantized_cdf_list = self.inner_module.quantized_cdf.tolist()
        self.cdf_length_list = self.inner_module.cdf_length.tolist()
        self.offset_list = self.inner_module.offset.tolist()

    def forward(self, x, means, scales):
        p = self.inner_module._likelihood(x, means=means, scales=scales)
        if self.inner_module.use_likelihood_bound:
            p = self.inner_module.likelihood_lower_bound(p)
        return p

    def encode_buffer_init(self):
        self.symbols_list = []
        self.indexes_list = []

    def encode_buffer_append(self, inputs, means, scales):
        assert isinstance(inputs, torch.Tensor), 'Inputs should be a torch.Tensor'
        assert means is None or isinstance(means, float) or (isinstance(means, torch.Tensor) and inputs.shape == means.shape), \
            f'Means should be a float or a tensor with the same shape as inputs'
        assert isinstance(scales, float) or (isinstance(scales, torch.Tensor) and inputs.shape == scales.shape), \
            f'Scales should be a float or a tensor with the same shape as inputs'
        indexes = self.inner_module.build_indexes(scales)
        symbols = self.inner_module.quantize(inputs, 'symbols', means)
        self.symbols_list.append(symbols.view(-1).cpu())
        self.indexes_list.append(indexes.view(-1).cpu())

    def encode_with_buffer(self):
        symbols = torch.concat(self.symbols_list).tolist()
        indexes = torch.concat(self.indexes_list).tolist()
        self.symbols_list = []
        self.indexes_list = []
        self.encoder.encode_with_indexes(symbols, indexes, self.quantized_cdf_list,
                                         self.cdf_length_list, self.offset_list)
        return self.encoder.flush()

    def decode_buffer_init(self, bs):
        self.decoder.set_stream(bs)

    def decode_with_buffer(self, inputs_ref, means, scales):
        assert isinstance(inputs_ref, torch.Tensor), 'Inputs should be a torch.Tensor'
        assert means is None or isinstance(means, float) or (isinstance(means, torch.Tensor) and inputs_ref.shape == means.shape), \
            f'Means should be a float or a tensor with the same shape as inputs_ref'
        assert isinstance(scales, float) or (isinstance(scales, torch.Tensor) and inputs_ref.shape == scales.shape), \
            f'Scales should be a float or a tensor with the same shape as inputs_ref'
        indexes = self.inner_module.build_indexes(scales)
        symbols = self.decoder.decode_stream(indexes.view(-1).tolist(), self.quantized_cdf_list,
                                             self.cdf_length_list, self.offset_list)
        return self.inner_module.dequantize(inputs_ref.new_tensor(symbols).view_as(inputs_ref), means)