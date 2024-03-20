import torch.nn as nn

from models import PredRNNpp_Model
from .predrnn import PredRNN


class PredRNNpp(PredRNN):
    r"""PredRNN++

    Implementation of `PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
    in Spatiotemporal Predictive Learning <https://arxiv.org/abs/1804.06300>`_.

    """

    def __init__(self, args, ds_config, base_criterion):
        PredRNN.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(self.args)

    def build_model(self, args):
        num_hidden = self.args.num_hidden
        return PredRNNpp_Model(num_hidden, args).to(self.device)

