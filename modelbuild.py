# -*- coding: utf-8 -*-
import os.path as osp
import torch
import torch.distributed as dist
import deepspeed
import math
from utils.utils import print_log, json2Parser, init_random_seed, set_seed, get_dist_info
from utils.ds_utils import get_train_ds_config
from core.optim_scheduler import get_optim_scheduler
import core.lossfun as lossfun
from methods import method_maps
import time
import logging
from utils.parser import default_parser



class modelbuild():
    def __init__(self, model_path, ds_args, mode = "train"):
        self.mode = mode
        self.model_path = model_path
        self.set_config(ds_args)
        self.build_method()


    def get_data(self):
        return self.args, self.method


    def build_method(self):
        self.init_lossfun()
        self.method = method_maps[self.args.method.lower()](self.args, self.ds_config, self.base_criterion)
        network = f"The neural network is created. Network type: {self.args.method.lower()}"
        if self.args.method.lower() == "simvp":
            network += f"({self.args.model_type})"
        print_log(network)
        self.init_optimizer()
        self.args.by_epoch = self.method.by_epoch
        self.method.model.eval()
        # setup ddp training
        if self.args.dist:
            self.method.init_distributed()


    def init_optimizer(self):
        """Create optimizer and scheduler."""
        (self.method.optimizer, self.method.scheduler, self.method.by_epoch) \
            = get_optim_scheduler(self.args, self.args.max_epoch, self.method.model, self.args.steps_per_epoch)
        print_log(f"The optimizer is created. Optimizer type: {self.args.optim}")
        print_log(f"The scheduler is created. Scheduler type: {self.args.sched}")
    

    def init_lossfun(self):
        """Setup base lossfun"""
        self.base_criterion = getattr(lossfun, self.args.lossfun)
        print_log(f"The base criterion is created. Base criterion type: {self.args.lossfun}")


    def init_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        prefix = 'train' if (self.mode == "train") else 'test'
        logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.model_path, '{}_{}.log'.format(prefix, timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')


    def initialize(self, args):
        """Initialize training environment.
        distributed by DeepSpeed.
        support both slurm and mpi or DeepSpeed to Initialize.
        """
        if args.local_rank == -1:
            self.init_logging()
            if torch.cuda.is_available():
                args.device = torch.device("cuda:0")
                print_log(f'Use non-distributed mode with GPU: {args.device}')
            else:
                args.device = torch.device("cpu")
                print_log(f'Use CPU')
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
            if args.rank == 0:
                self.init_logging()
            print_log(f'Use distributed mode with GPUs, world_size: {args.world_size}')
        
        if args.dist:
            seed = init_random_seed(args.seed)
            seed = seed + dist.get_rank() if args.diff_seed else seed
        else:
            seed = args.seed
        set_seed(seed)
        return args


    def set_config(self, ds_args):
        """Setup config"""
        # read config
        setting_path = osp.join(self.model_path, 'checkpoints', f'settings.json')
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
        args.in_shape = [args.data_previous, args.data_select_num,
                          args.data_range[0][1]-args.data_range[0][0],
                          args.data_range[1][1]-args.data_range[1][0]]
        args.out_shape = [args.data_after, args.data_select_num,
                          args.data_range[0][1]-args.data_range[0][0],
                          args.data_range[1][1]-args.data_range[1][0]]
        args.data_use = [args.data_type[i] for i in args.data_select]
        
        args = self.initialize(args)
        args.batch_size = args.per_device_train_batch_size * args.world_size
        trainlen = int((1 - args.valid_ratio) * int(args.data_num - args.data_previous - args.data_after))
        args.steps_per_epoch = math.ceil(trainlen/args.world_size/args.per_device_train_batch_size)
        ds_steps_per_print = args.max_epoch * args.steps_per_epoch + 1 # close ds step per print

        ds_config = get_train_ds_config(args, ds_steps_per_print)
        self.args=args
        self.ds_config = ds_config


