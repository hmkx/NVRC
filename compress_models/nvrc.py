"""
NVRC codec
"""
from .utils import *
from .codec_base import *
from entropy_models.entropy_model_base import (
    FullPrecisionModel, HalfPrecisionModel, _EntropyModelBase, EntropyModel
)
from entropy_models.weight_entropy_model import (
    WeightEntropyModel, SingleAxisEntropyModel, DoubleAxisEntropyModel,
)
from entropy_models.grid_entropy_model import (
    GridEntropyModel, ARGridEntropyModel
)


@dataclass
class EMConfigBase:
    em: str = 'em'
    em_dist: str = 'gaussian'
    rate_quant: str = 'soft-round-noise'
    dist_quant: str = 'soft-round-noise'
    quant_mean: bool = False
    learned_quant: bool = True
    learned_em: bool = True


@dataclass
class GridEMConfig(EMConfigBase):
    blk_size: List[int] = field(default_factory=lambda: [20, 5, 5])
    per_blk: bool = True
    channels: int = 4
    depth: int = 3
    kernel_size: int = 3
    norm: str = 'layernorm'
    bias: bool = True
    act: str = 'gelu'
    depthwise: bool = False


@dataclass
class WeightEMConfig(EMConfigBase):
    blk_size: List[int] = field(default_factory=lambda: [16, 32])
    per_blk: bool = True
    per_axis: bool = True


@dataclass
class MetaEMConfig(EMConfigBase):
    pass


@dataclass
class OptimizeConfig:
    optimize_rate: bool = True
    init_bit_width: int = -1
    soft_round_temp: List[float] = field(default_factory=lambda: [0.3, 0.1])
    kum_noise: List[float] = field(default_factory=lambda: [2.0, 1.0])
    quant_noise: List[float] = field(default_factory=lambda: [0.0, 0.9])


@dataclass
class NVRCConfig:
    grid_em: GridEMConfig = field(default_factory=GridEMConfig)
    weight_em: WeightEMConfig = field(default_factory=WeightEMConfig)
    grid_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    weight_meta_em: MetaEMConfig = field(default_factory=MetaEMConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    def __post_init__(self):
        self.grid_em = GridEMConfig(**self.grid_em) \
                       if isinstance(self.grid_em, dict) else self.grid_em
        self.weight_em = WeightEMConfig(**self.weight_em) \
                         if isinstance(self.weight_em, dict) else self.weight_em
        self.grid_meta_em = MetaEMConfig(**self.grid_meta_em) \
                            if isinstance(self.grid_meta_em, dict) else self.grid_meta_em
        self.weight_meta_em = MetaEMConfig(**self.weight_meta_em) \
                              if isinstance(self.weight_meta_em, dict) else self.weight_meta_em
        self.optimize = OptimizeConfig(**self.optimize) \
                        if isinstance(self.optimize, dict) else self.optimize


class NVRC(CodecBase):
    """
    NVRC codec
    """
    def __init__(self, logger, model, config):
        super().__init__()
        self.logger = logger
        self.config = copy.deepcopy(config)

        logger.info(f'NVRC:')
        logger.info(f'     Optimize rate - {self.config.optimize.optimize_rate}  Init bit width - {self.config.optimize.init_bit_width}')
        logger.info(f'     Soft-rounding temp - {self.config.optimize.soft_round_temp}  Kumaraswamy - {self.config.optimize.kum_noise}')
        logger.info(f'     QuantNoise - {self.config.optimize.quant_noise}')
        logger.info(f'     Cache enabled - {self.cache_enabled}')

        # INR
        self.model = model

        # INR param model
        init_params = self.model.get_params_recursive()
        self.param_model = ParamModel(logger, init_params)

        params = self.param_model()
        self._set_split_params(init_params)
        grids, weights, _ = self._split_params(params)

        # Grid em
        self._build_grid_em(logger, grids=grids, config=self.config.grid_em)
        self.grid_em.init_func_modules()
        self.grid_em_param_model = ParamModel(
            logger, self.grid_em.get_params_recursive()
        )

        # Weight em
        self._build_weight_em(logger, weights=weights, config=self.config.weight_em)
        self.weight_em.init_func_modules()
        self.weight_em_param_model = ParamModel(
            logger, self.weight_em.get_params_recursive()
        )

        # Grid meta
        self._build_grid_meta_em(logger, grid_em_weights=self.grid_em_param_model(),
                                 config=self.config.grid_meta_em)
        self.grid_meta_em.init_func_modules()
        self.grid_meta_em_param_model = ParamModel(
            logger, self.grid_meta_em.get_params_recursive()
        )

        # Weight meta
        self._build_weight_meta_em(logger, weight_em_weights=self.weight_em_param_model(),
                                   config=self.config.weight_meta_em)
        self.weight_meta_em.init_func_modules()
        self.weight_meta_em_param_model = ParamModel(
            logger, self.weight_meta_em.get_params_recursive()
        )

        # Meta
        self._build_meta_em(logger, inputs={})
        self.meta_em.init_func_modules()

    def _build_grid_em(self, logger, grids, config: GridEMConfig):
        logger.info(f'     Grid EM:')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')
        logger.info(f'          Blk size - {config.blk_size}  Per blk - {config.per_blk}')
        logger.info(f'          Channels - {config.channels}  Depth - {config.depth}')
        logger.info(f'          Kernel size - {config.kernel_size}  Norm - {config.norm}  Bias - {config.bias}  Act - {config.act}')
        logger.info(f'          Depthwise - {config.depthwise}')

        try:
            em_cls = {
                'em': EntropyModel,
                'blk': GridEntropyModel,
                'ar': ARGridEntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]
        except KeyError:
            raise ValueError(f'Unknown grid entropy model {config.em}')
        if issubclass(em_cls, _EntropyModelBase):
            self.grid_em = em_cls(logger, name='grid_em', inputs=grids, config=config)
        else:
            self.grid_em = em_cls(logger, name='grid_em', inputs=grids)

    def _build_weight_em(self, logger, weights, config: WeightEMConfig):
        logger.info(f'     Weight EM:')
        logger.info(f'          EM - {config.em}  Dist - {config.em_dist}  Rate quant - {config.rate_quant}')
        logger.info(f'          Blk size - {config.blk_size}  Per blk - {config.per_blk}  Per axis - {config.per_axis}')

        em_cls = {
            'em': EntropyModel,
            'blk': WeightEntropyModel,
            's-axis': SingleAxisEntropyModel,
            'd-axis': DoubleAxisEntropyModel,
            'fp32': FullPrecisionModel,
            'fp16': HalfPrecisionModel,
        }[config.em]

        if issubclass(em_cls, _EntropyModelBase):
            self.weight_em = em_cls(logger, name='weight_em', inputs=weights, config=config)
        else:
            self.weight_em = em_cls(logger, name='weight_em', inputs=weights)

    def _build_grid_meta_em(self, logger, grid_em_weights, config: MetaEMConfig):
        logger.info(f'     Grid meta - {config.em}')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')

        if grid_em_weights:
            em_cls = {
                'em': EntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.grid_meta_em = em_cls(logger, name='grid_meta_em', inputs=grid_em_weights, config=config)
            else:
                self.grid_meta_em = em_cls(logger, name='grid_meta_em', inputs=grid_em_weights)
        else:
            self.grid_meta_em = FullPrecisionModel(logger, name='grid_meta_em', inputs={})

    def _build_weight_meta_em(self, logger, weight_em_weights, config: WeightEMConfig):
        logger.info(f'     Weight meta - {config.em}')
        logger.info(f'          EM - {config.em}  EM-dist - {config.em_dist}  Rate quant - {config.rate_quant}  Dist quant - {config.dist_quant}')
        logger.info(f'          Use quant mean - {config.quant_mean}  Learned quant - {config.learned_quant}  Learned em - {config.learned_em}')

        if weight_em_weights:
            em_cls = {
                'em': EntropyModel,
                'fp32': FullPrecisionModel,
                'fp16': HalfPrecisionModel,
            }[config.em]

            if issubclass(em_cls, _EntropyModelBase):
                self.weight_meta_em = em_cls(logger, name='weight_meta_em', inputs=weight_em_weights,
                                             config=config)
            else:
                self.weight_meta_em = em_cls(logger, name='weight_meta_em', inputs=weight_em_weights)
        else:
            self.weight_meta_em = FullPrecisionModel(logger, name='weight_meta_em', inputs={})

    def _build_meta_em(self, logger, inputs):
        self.meta_em = FullPrecisionModel(logger, name='meta_em', inputs=inputs)

    def _set_split_params(self, params):
        self.param_split_types = {}

        for k, v in params.items():
            if v is not None:
                assert isinstance(v, nn.Parameter), 'Inputs should be nn.Parameter.'
                if is_grid_param(v):
                    self.param_split_types[k] = 'grid'
                elif v.ndim >= 2:
                    self.param_split_types[k] = 'weight'
                else:
                    self.param_split_types[k] = 'others'
            else:
                self.param_split_types[k] = 'others'
        
    def _split_params(self, params):
        # Split parameters into grids and weights
        grids = {}
        weights = {}
        others = {}

        for k, v in params.items():
            param_split_type = self.param_split_types[k]
            if param_split_type == 'grid':
                grids[k] = v
            elif param_split_type == 'weight':
                weights[k] = v
            else:
                others[k] = v

        return grids, weights, others

    def _forward_rates_params(self, x=None, forward_rates=True, forward_params=True):
        # Grid meta params
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_meta_em_params_rate, grid_meta_em_params_q = self.meta_em(
            grid_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        # Grid em params
        grid_em_params = self.grid_em_param_model()
        grid_em_params_rate, grid_em_params_q = self.grid_meta_em(
            grid_em_params, grid_meta_em_params_q, forward_rates=forward_rates, forward_params=True
        )

        # Weight meta params
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_meta_em_params_rate, weight_meta_em_params_q = self.meta_em(
            weight_meta_em_params, forward_rates=forward_rates, forward_params=True
        )

        # Weight em params
        weight_em_params = self.weight_em_param_model()
        weight_em_params_rate, weight_em_params_q = self.weight_meta_em(
            weight_em_params, weight_meta_em_params_q, forward_rates=forward_rates, forward_params=True
        )

        # Model params
        params = self.param_model()
        grids, weights, others = self._split_params(params)
        grids_rate, grids_q = self.grid_em(
            grids, grid_em_params_q, forward_rates=forward_rates, forward_params=forward_params
        )
        weights_rate, weights_q = self.weight_em(
            weights, weight_em_params_q, forward_rates=forward_rates, forward_params=forward_params
        )
        others_rate, others_q = self.meta_em(
            others, forward_rates=forward_rates, forward_params=forward_params
        )

        # Total rate
        if forward_rates:
            rates = {
                k: torch.tensor(v, device=next(iter(params.values())).device) if not isinstance(v, torch.Tensor) else v
                for k, v in {
                    'grid': grids_rate, 'weight': weights_rate,
                    'grid_em_params': grid_em_params_rate, 'weight_em_params': weight_em_params_rate,
                    'grid_meta_em_params': grid_meta_em_params_rate,
                    'weight_meta_em_params': weight_meta_em_params_rate,
                    'others': others_rate
                }.items()
            }
        else:
            rates = None

        #  All params
        if forward_params:
            params_q = {**grids_q, **weights_q, **others_q} \
                       if self.config.optimize.optimize_rate else {**grids, **weights, **others}
        else:
            params_q = None

        return rates, params_q

    def _get_rate(self, rates, reduce_rates):
        if reduce_rates:
            return sum(rates.values())
        else:
            return rates

    def _get_rate_loss(self, rates, reduce_rates):
        if reduce_rates:
            return rates['grid'] + rates['weight'] \
                   + rates['grid_em_params'] + rates['weight_em_params'] \
                   + rates['grid_meta_em_params'] + rates['weight_meta_em_params'] \
                   + rates['others']
        else:
            return {
                'grid': rates['grid'],
                'weight': rates['weight'],
                'grid_em_params': rates['grid_em_params'],
                'weight_em_params': rates['weight_em_params'],
                'grid_meta_em_params': rates['grid_meta_em_params'],
                'weight_meta_em_params': rates['weight_meta_em_params'],
                'others': rates['others']
            }

    def get_num_params(self):
        grid_meta_em_params = self.grid_meta_em_param_model()
        grid_em_params = self.grid_em_param_model()
        weight_meta_em_params = self.weight_meta_em_param_model()
        weight_em_params = self.weight_em_param_model()
        params = self.param_model()
        grids, weights, others = self._split_params(params)
        num_params = {
            k: sum([p.numel()  if p is not None else 0 for p in v.values()])
            for k, v in {
                'grid': grids, 'weight': weights,
                'grid_em_params': grid_em_params, 'weight_em_params': weight_em_params,
                'grid_meta_em_params': grid_meta_em_params, 'weight_meta_em_params': weight_meta_em_params,
                'others': others,
            }.items()
        }
        return num_params

    def set_epoch(self, epoch, num_epochs):
        epoch_ratio = epoch / (num_epochs - 1) if num_epochs > 1 else 0.
        init_bit_width = self.config.optimize.init_bit_width \
                         if (not self.config.optimize.optimize_rate or epoch == 0) and \
                            self.config.optimize.init_bit_width != -1 else None
        soft_round_temp = self.config.optimize.soft_round_temp[0] + \
                          (self.config.optimize.soft_round_temp[1] - self.config.optimize.soft_round_temp[0]) * \
                          epoch_ratio
        kum_noise = self.config.optimize.kum_noise[0] + \
                    (self.config.optimize.kum_noise[1] - self.config.optimize.kum_noise[0]) * epoch_ratio
        quant_noise = self.config.optimize.quant_noise[0] + \
                      (self.config.optimize.quant_noise[1] - self.config.optimize.quant_noise[0]) * epoch_ratio

        # Model params
        params = self.param_model()
        grid, weight, _ = self._split_params(params)
        if isinstance(self.grid_em, _EntropyModelBase):
            self.grid_em.set_params(grid, init_bit_width,
                                    soft_round_temp, kum_noise, quant_noise,
                                    self.grid_em_param_model())
        if isinstance(self.weight_em, _EntropyModelBase):
            self.weight_em.set_params(weight, init_bit_width,
                                      soft_round_temp, kum_noise, quant_noise,
                                      self.weight_em_param_model())

        # Grid em params
        if isinstance(self.grid_meta_em, _EntropyModelBase):
            self.grid_meta_em.set_params(self.grid_em_param_model(), init_bit_width, 
                                         soft_round_temp, kum_noise, quant_noise,
                                         self.grid_meta_em_param_model())

        # Weight em params
        if isinstance(self.weight_meta_em, _EntropyModelBase):
            self.weight_meta_em.set_params(self.weight_em_param_model(), init_bit_width,
                                           soft_round_temp, kum_noise, quant_noise,
                                           self.weight_meta_em_param_model())

    def set_iter(self, iter, num_iters):
        return self.set_epoch(iter, num_iters)

    def set_zero(self):
        for model in [self.param_model, self.grid_em_param_model, self.weight_em_param_model,
                      self.grid_meta_em_param_model, self.weight_meta_em_param_model,
                      self.model, self.grid_em, self.weight_em, self.grid_meta_em, self.weight_meta_em]:
            for param in model.parameters():
                param.data.zero_()

    def compress(self, bs):
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.compress(bs, grid_meta_em_params)

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.compress(bs, weight_meta_em_params)

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.compress(bs, grid_em_params, grid_meta_em_params)

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.compress(bs, weight_em_params, weight_meta_em_params)

                # Model params
                params = self.param_model()
                grid, weight, others = self._split_params(params)
                bs, _, _ = self.grid_em.compress(bs, grid, grid_em_params)
                bs, _, _ = self.weight_em.compress(bs, weight, weight_em_params)
                bs, _, _ = self.meta_em.compress(bs, others)

        return bs

    def decompress(self, bs):
        self.eval()

        with DeterministicContext():
            with torch.no_grad():
                # Grid meta params
                grid_meta_em_params = self.grid_meta_em_param_model()
                bs, _, grid_meta_em_params = self.meta_em.decompress(bs, grid_meta_em_params)

                # Weight meta params
                weight_meta_em_params = self.weight_meta_em_param_model()
                bs, _, weight_meta_em_params = self.meta_em.decompress(bs, weight_meta_em_params)

                # Grid em params
                grid_em_params = self.grid_em_param_model()
                bs, _, grid_em_params = self.grid_meta_em.decompress(bs, grid_em_params, grid_meta_em_params)

                # Weight em params
                weight_em_params = self.weight_em_param_model()
                bs, _, weight_em_params = self.weight_meta_em.decompress(bs, weight_em_params, weight_meta_em_params)

                # Model params
                params = self.param_model()
                grid, weight, others = self._split_params(params)
                bs, _, grid = self.grid_em.decompress(bs, grid, grid_em_params)
                bs, _, weight = self.weight_em.decompress(bs, weight, weight_em_params)
                bs, _, others = self.meta_em.decompress(bs, others)

                self.grid_meta_em_param_model.load(grid_meta_em_params)
                self.weight_meta_em_param_model.load(weight_meta_em_params)
                self.grid_em_param_model.load(grid_em_params)
                self.weight_em_param_model.load(weight_em_params)
                self.param_model.load({**grid, **weight, **others})

        return bs

    def get_rate_stat(self):
        with torch.no_grad():
            rates, _ = self._forward_rates_params(x=None, forward_rates=True, forward_params=False)
            rates = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in rates.items()}
            num_params = self.get_num_params()
            num_params = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in num_params.items()}
            bits_per_params = {k: rates[k] / num_params[k] if num_params[k] > 0 else math.nan for k in rates.keys()}
        return rates, num_params, bits_per_params


def build_compress_model(logger, model, cfg):
    # Build NVRC model
    return NVRC(logger, model, config=NVRCConfig(**cfg))