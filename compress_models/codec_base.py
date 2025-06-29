"""
Base class for compression models
"""
from .utils import *


class CodecBase(FuncModule):
    """
    Codec base class
    """
    cache_enabled = True
    cached = False
    rates_cache = None
    params_cache = None
    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        """
        Forward parameters and rates
        """
        raise NotImplementedError

    def _get_rate(self, rates):
        """
        Get rate
        """
        raise NotImplementedError

    def _get_rate_loss(self, rates):
        """
        Get rate loss
        """
        raise NotImplementedError

    def forward(self, x, compute_outputs=True, compute_rates=True, compute_params=False, reduce_rates=True):
        """
        Forward pass
        """
        assert compute_outputs or compute_rates or compute_params, \
            'At least one of get_outputs and get_rates should be True'

        all_outputs = []

        # Compute params and rates
        if not self.cached:
            rates, params = self._forward_rates_params(x, forward_rates=compute_rates,
                                                       forward_params=compute_outputs or compute_params)
        else:
            rates = self.rates_cache
            params = self.params_cache

        # Outputs
        if compute_outputs:
            outputs = self.model(x, params)
            all_outputs.append(outputs)

        # Rates
        if compute_rates:
            rate = self._get_rate(rates, reduce_rates)
            rate_loss = self._get_rate_loss(rates, reduce_rates)
            all_outputs.append(rate)
            all_outputs.append(rate_loss)

        # Params
        if compute_params:
            all_outputs.append(params)
        
        return tuple(all_outputs) if len(all_outputs) > 1 else all_outputs[0]

    def load_cache(self):
        """
        Load parameters and rate cache
        """
        if self.cache_enabled:
            with torch.no_grad():
                self.rates_cache, _, self.params_cache = self(None, compute_outputs=False, compute_rates=True,
                                                              compute_params=True, reduce_rates=False)
                self.cached = True

    def clear_cache(self):
        """
        Clear parameters and rate cache
        """
        self.rates_cache, self.params_cache = None, None
        self.cached = False

    def compress(self, bs):
        """
        Compress
        """
        raise NotImplementedError

    def decompress(self, bs):
        """
        Decompress
        """
        raise NotImplementedError

    def get_rate_stat(self):
        """
        Get rate statistics
        """
        raise NotImplementedError

    def set_epoch(self, epoch, num_epochs):
        """
        Set epoch
        """
        raise NotImplementedError
    
    def set_iter(self, iter, num_iters):
        """
        Set iter
        """
        raise NotImplementedError