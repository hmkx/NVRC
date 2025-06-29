from utils import *
from models.layers import *


class DeterministicContext:
    """
    Deterministic context
    """
    def __enter__(self):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *args):
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


class ParamModel(nn.Module):
    """
    ParamModel    
    """
    def __init__(self, logger, params, requires_grad=True):
        super().__init__()
        # Params
        self.params = nn.ParameterDict(params)

        # Set requires_grad
        self.set_param_requires_grad(requires_grad)

    def set_param_requires_grad(self, requires_grad):
        for param in self.params.values():
            if param is not None:
                param.requires_grad = requires_grad

    def load(self, params):
        self.params.load_state_dict(params)

    def forward(self):
        return self.params