# -*- coding: utf-8 -*-
import numpy as np
import os
import json
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
nx = 1181
ny = 220
data_type_num=4
data_num = 1001
data_path = "./../cfd-data-1001"
setting_path = f"./{data_num}.json"

def save_json(data, data_path):
    """Save json data"""
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_and_process_data(j):
    """Load and process data for a given index"""
    flow_data_path = os.path.join(data_path, f"flowxy-{j:04d}.dat")
    print(flow_data_path)
    data = np.loadtxt(flow_data_path, skiprows=2)
    processed_data = np.zeros((data_type_num, ny * nx))
    for i in range(data_type_num):
        processed_data[i] = data[:, i + 2]
    return processed_data

def main():
    ## model path
    input_data_list = np.zeros((data_type_num, data_num, ny * nx))
    # Use ThreadPoolExecutor to parallelize data loading and processing
    with ThreadPoolExecutor() as executor:
        # Map the function over the range of file indices
        results = list(executor.map(load_and_process_data, range(1, 1 + data_num)))
    # Aggregate the results
    for j, result in enumerate(results):
        input_data_list[:, j] = result
    
    mean = []
    std = []
    max = []
    min = []
    for i in range(data_type_num):
        mean.append(np.mean(input_data_list[i]))
        std.append(np.std(input_data_list[i]))
        max.append(input_data_list[i].max())
        min.append(input_data_list[i].min())
    
    args = edict({})
    
    args.data_mean = mean
    args.data_std = std
    args.data_max = max
    args.data_min = min
    save_json(args,setting_path)
    print(f"Successful save the norm data to {setting_path}")

if __name__ == '__main__':
    main()