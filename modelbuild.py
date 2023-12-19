# -*- coding: utf-8 -*-
import os
from modeldefine import *
import torch
import torch.distributed as dist
import deepspeed
import math
from utils.utils import print_rank_0, json2Parser, init_random_seed, set_seed, get_dist_info
from utils.ds_utils import get_train_ds_config
from core.optim_scheduler import get_optim_scheduler
import core.lossfun as lossfun
from utils.parser import default_parser
from models import model_maps

from easydict import EasyDict as edict

def initialize(args):
    """Initialize training environment.
       distributed by DeepSpeed.
       support both slurm and mpi or DeepSpeed to Initialize.
    """
    if args.local_rank == -1:
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")
            print_rank_0(f'Use non-distributed mode with GPU: {args.device}')
        else:
            args.device = torch.device("cpu")
            print_rank_0(f'Use CPU')
        args.rank = 0
        args.world_size = 1
        args.dist = False
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
        args.rank, args.world_size = get_dist_info()
        args.dist = True
        print_rank_0(f'Use distributed mode with GPUs, world_size: {args.world_size}')
    
    if args.dist:
        seed = init_random_seed(args.seed)
        seed = seed + dist.get_rank() if args.diff_seed else seed
    else:
        seed = args.seed
    set_seed(seed)
    return args


class modelbuild():
    def __init__(self, model_path, ds_args):
        self.set_config(model_path, ds_args)
        self.build_model(self.args)
        self.init_optimizer(self.args)
        self.init_lossfun(self.args)


    def get_data(self):
        return self.args, self.ds_config, self.net, self.optimizer, self.scheduler, self.base_criterion


    def build_model(self, args):
        """Create a neural network."""
        model_config = edict(args.model_config)
        model_config.in_shape = args.data_previous, args.data_select_num, \
                                args.data_height, args.data_width
        self.args.in_shape = model_config.in_shape
        net = model_maps[args.method.lower()]
        self.net = net(**model_config).to(args.device)
        print_rank_0(f"The neural network is created. Network type: {args.method.lower()}")


    def init_optimizer(self, args):
        """Create optimizer and scheduler."""
        (self.optimizer, self.scheduler, self.args.by_epoch) \
            = get_optim_scheduler(args, args.max_epoch, self.net, args.steps_per_epoch)
        print_rank_0(f"The optimizer is created. Optimizer type: {args.optim}")
        print_rank_0(f"The scheduler is created. Scheduler type: {args.sched}")
    

    def init_lossfun(self, args):
        """Setup base lossfun"""
        self.base_criterion = getattr(lossfun, args.lossfun)
        print_rank_0(f"The base criterion is created. Base criterion type: {args.lossfun}")


    def set_config(self, model_path, ds_args):
        """Setup config"""
        # read config
        print_rank_0("\n")
        setting_path = os.path.join(model_path, 'checkpoints', f'settings.json')
        args = json2Parser(setting_path)
        default_values = default_parser()
        for attribute in default_values.keys():
            for key in default_values[attribute].keys():
                if key not in args[attribute].keys():
                    args[attribute][key] = default_values[attribute][key]
        
        for key in list(args.keys()):
            args.update(args[key])

        args.total_length = args.data_previous + args.data_after
        args.data_type_num = len(args.data_type)
        args.data_select_num = len(args.data_select)
        args.local_rank = ds_args.local_rank
        args.data_shape = [args.data_type_num, args.data_height, args.data_width]
        args.out_shape = [args.data_select_num,
                            args.data_range[0][1]-args.data_range[0][0],
                            args.data_range[1][1]-args.data_range[1][0]]
        args.data_use = [args.data_type[i] for i in args.data_select]
        
        args = initialize(args)
        args.batch_size = args.per_device_train_batch_size * args.world_size
        trainlen = int((1 - args.valid_ratio) * int(args.data_num - args.data_previous - args.data_after))
        args.steps_per_epoch = math.ceil(trainlen/args.world_size/args.per_device_train_batch_size)

        if args.print_ds_output:
            steps_per_print = args.steps_per_epoch
        else:
            steps_per_print = args.max_epoch * args.steps_per_epoch + 1

        ds_config = get_train_ds_config(args, steps_per_print)
        self.args=args
        self.ds_config = ds_config

