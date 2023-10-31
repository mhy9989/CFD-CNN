# -*- coding: utf-8 -*-
import torch
import numpy as np
import json
import os
from utils.utils import json2Parser, save_json

def main():
    modelname = 'CFD_Conv1001_5to1'
    ## model path
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, 'Model', f'{modelname}')
    setting_path = os.path.join(model_path, 'checkpoint', f'settings.json')
    args = json2Parser(setting_path)
    nx = args.data_width
    ny = args.data_height
    data_type_num = len(args.data_type)
    input_data_list = np.zeros((data_type_num,args.data_num, ny*nx))
    for j in range(1, 1+args.data_num):
        data_path = os.path.join(args.data_path , f"flowxy-{j:04d}.dat")
        print(data_path)
        data = np.loadtxt(data_path,skiprows=2)
        for i in range(data_type_num):
            input_data_list[i, j-1] = data[:, i+2]
    
    mean = []
    std = []
    max = []
    min = []
    for i in range(data_type_num):
        mean.append(np.mean(input_data_list[i]))
        std.append(np.std(input_data_list[i]))
        max.append(input_data_list[i].max())
        min.append(input_data_list[i].min())
    
    args.data_mean = mean
    args.data_std = std
    args.data_max = max
    args.data_min = min
    save_json(args,setting_path)
    print(f"Successful save the norm data to {setting_path}")

if __name__ == '__main__':
    main()