
# -*- coding: utf-8 -*-
import torch.distributed as dist

import json
from easydict import EasyDict as edict


def print_rank_0(message):
    """Only output in root process or single process
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def save_json(data,data_path):
    """Save json data
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            with open(data_path, 'w') as json_file:
                json.dump(data,json_file,indent=4)
    else:
        with open(data_path, 'w') as json_file:
            json.dump(data,json_file,indent=4)


def json2Parser(json_path):
    """Load json and return a parser-like object
    Parameters
    ----------
    json_path : str
        The json file path.
    
    Returns
    -------
    args : easydict.EasyDict
        A parser-like object.
    """
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)


def get_all_reduce_mean(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor

