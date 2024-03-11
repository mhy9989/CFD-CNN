import torch.nn as nn

from models import E3DLSTM_Model
from .predrnn import PredRNN


class E3DLSTM(PredRNN):
    r"""E3D-LSTM

    Implementation of `EEidetic 3D LSTM: A Model for Video Prediction and Beyond
    <https://openreview.net/forum?id=B1lKS2AqtX>`_.

    """

    def __init__(self, args, ds_config, base_criterion):
        PredRNN.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(self.args)

    def build_model(self, args):
        num_hidden = self.args.num_hidden
        return E3DLSTM_Model(num_hidden, args).to(self.device)
