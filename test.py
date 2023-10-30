# -*- coding: utf-8 -*-
from modeltrain import modeltrain
from modelbuild import modelbuild
import os
import argparse
import deepspeed

def add_argument():
    parser = argparse.ArgumentParser(description='CFD-CNN')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")
    parser.add_argument('--seed',
                        type=int,
                        default=2023,
                        help="random seed for initialization")
    parser.add_argument('--zero_stage',
                        type=int,
                        default=0,
                        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():
    ## model name
    modelname = 'CFD_ConvLSTM_1to1'
    ## model path
    dir_path = os.path.dirname(os.path.abspath(__file__))

    ds_args = add_argument()
    model_path = os.path.join(dir_path, 'Model', f'{modelname}')
    model_data = modelbuild().loadsetting(model_path,ds_args)
    model = modeltrain(model_data)
    #model.train_CFD(model_path)
    model.test_CFD(model_path,-1)
    #model.summary_model((1,220,1181),"cpu")



if __name__ == '__main__':
    main()