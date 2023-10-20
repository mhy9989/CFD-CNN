# -*- coding: utf-8 -*-
import os
import numpy as np
from modeldefine import *
import torch
from torch import nn,optim
import torch.backends.cudnn as cudnn
import random
import torch.distributed as dist
import deepspeed
import math
from utils.utils import print_rank_0, json2Parser
from utils.ds_utils import *

def Initialize(args):
    """Initialize training environment.
    distributed by DeepSpeed.
    support both slurm and mpi or DeepSpeed to Initialize.
    """
    cudnn.enabled = True   # If use DCU, make it False 
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.local_rank == -1:
        if torch.cuda.is_available():
            args.device = torch.device("cuda:0")
        else:
            args.device = torch.device("cpu")
        args.global_rank = 0
        args.world_size = 1
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
        args.global_rank = dist.get_rank()
        args.world_size = dist.get_world_size()

    return args

class modelbuild():
    def __init__(self) -> None:
        pass

    def buildModel(self, args):
        """Create a neural network."""
        act_funs = {
            'Relu': nn.ReLU(),
            'Gelu': nn.GELU(),
            'Tanh': nn.Tanh(),
            'LeakyReLU' : nn.LeakyReLU(),
            'Sigmoid': nn.Sigmoid()
        }
        net = {
            'CFD_ConLSTM': CFD_ConLSTM,
        }
        self.net = net[args.net_type](act_funs[args.actfun],args.data_out_shape,args.device)
        print_rank_0(f"The neural network is created. Network type: {args.net_type}")

    def setModel(self, args):
        """Set the neural network."""
        optim_hparas = {
            'lr': args.learnrate,
        }
        self.optimizer= getattr(optim, args.optim)(
        self.net.parameters(), **optim_hparas)
        print_rank_0(f"The optimizer  is created. Optimizer type: {args.optim}")
        # Setup Lossfun
        loss_funs = {
            'MAE':nn.L1Loss(),
            'MSE':nn.MSELoss(),
            'SmoothL1Loss':nn.SmoothL1Loss(),
            'CEL': nn.CrossEntropyLoss(),
            'Log': nn.LogSoftmax()
        }
        self.criterion=loss_funs[args.lossfun]
        print_rank_0(f"The criterion  is created. criterion type: {args.lossfun}")

    def loadsetting(self, model_path,ds_args):
        """Load setting and build model"""
        setting_path = os.path.join(model_path, 'checkpoint', f'settings.json')
        args = json2Parser(setting_path)
        args.local_rank = ds_args.local_rank
        args.data_shape = [args.data_type_num,
                            args.data_height,
                            args.data_width]
        args.data_out_shape = [args.data_type_num,
                            args.data_range[0][1]-args.data_range[0][0],
                            args.data_range[1][1]-args.data_range[1][0]
                          ]
        args = Initialize(args)
        trainlen = int((1 - args.valid_ratio) * (args.data_num - args.data_delt -1))
        args.steps_per_epoch = math.ceil(trainlen/args.world_size)
        if args.print_ds_output:
            steps_per_print = args.steps_per_epoch
        else:
            steps_per_print = args.max_epoch * args.steps_per_epoch

        ds_config = get_train_ds_config(offload=ds_args.offload, 
                                        stage=ds_args.zero_stage,
                                        steps_per_print=steps_per_print)
        
        ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
        
        ds_config['train_batch_size'] = \
            args.per_device_train_batch_size * \
            args.world_size * \
            args.gradient_accumulation_steps


        if dist.is_initialized():
            dist.barrier()
        self.buildModel(args)
        self.setModel(args)
        self.args=args
        self.ds_config = ds_config

        return (self.args,self.ds_config,self.net,self.optimizer,self.criterion)


