import torch.nn as nn

from models import ConvLSTM_Model
from .predrnn import PredRNN


class ConvLSTM(PredRNN):
    r"""ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Notice: ConvLSTM requires `find_unused_parameters=True` for DDP training.
    """

    def __init__(self, args, ds_config, base_criterion):
        PredRNN.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(self.args)

    def build_model(self, args):
        num_hidden = self.args.num_hidden
        return ConvLSTM_Model(num_hidden, args).to(self.device)

