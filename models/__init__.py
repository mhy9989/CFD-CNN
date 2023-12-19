from .convlstm_model import ConvLSTM_Model
from .crevnet_model import CrevNet_Model
from .e3dlstm_model import E3DLSTM_Model
from .mau_model import MAU_Model
from .mim_model import MIM_Model
from .phydnet_model import PhyDNet_Model
from .predrnn_model import PredRNN_Model
from .predrnnpp_model import PredRNNpp_Model
from .predrnnv2_model import PredRNNv2_Model
from .simvp_model import SimVP_Model
from .dmvfn_model import DMVFN_Model

model_maps = {
    # 'convlstm': ConvLSTM,
    # 'crevnet': CrevNet,
    # 'e3dlstm': E3DLSTM,
    # 'mau': MAU,
    # 'mim': MIM,
    # 'phydnet': PhyDNet,
    # 'prednet': PredNet,
    # 'predrnn': PredRNN,
    # 'predrnnpp': PredRNNpp,
    # 'predrnnv2': PredRNNv2,
    'simvp': SimVP_Model,
    'tau': SimVP_Model,
    # 'dmvfn': DMVFN
}

__all__ = [
    'ConvLSTM_Model', 'CrevNet_Model', 'E3DLSTM_Model', 'MAU_Model', 'MIM_Model',
    'PhyDNet_Model', 'PredRNN_Model', 'PredRNNpp_Model', 'PredRNNv2_Model', 'SimVP_Model',
    'DMVFN_Model'
]