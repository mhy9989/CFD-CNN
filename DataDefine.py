# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from utils.utils import print_rank_0, save_json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from scipy.stats import yeojohnson  
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_datloader(args, mode = "train", test_num = 0):
    """Generate dataloader"""
    dataset = CFD_Dataset(
        args.data_path,
        args.coordinate_path,
        args.data_select,
        args.data_mean,
        args.data_std,
        args.data_max,
        args.data_min,
        args.data_range,
        args.data_shape,
        args.data_num,
        args.data_previous,
        args.data_after)
    data_scaler_list = dataset.scaler_list
    print_rank_0(f"\nLength of all dataset: {len(dataset)}")
    # Split dataset into training dataset, validation dataset and test_dataset
    # The last line of data is test data
    
    if mode != "train":
        if test_num < 0:
            test_num += len(dataset)
        test_dataset = dataset[test_num]
        return test_dataset, data_scaler_list, dataset.x_site_matrix, dataset.y_site_matrix
    else:
        test_num = len(dataset) -1
        test_dataset = dataset[test_num]
    
    # Reset the length of dataset, del last line
    dataset.custom_length -= 1
    trainlen = int((1 - args.valid_ratio) * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    train_dataset, valid_dataset = random_split(dataset, lengths)
    
    print_rank_0(f"Length of input dataset: {len(dataset)}")
    print_rank_0(f"Length of train_dataset: {len(train_dataset)}")
    print_rank_0(f"Length of valid_dataset: {len(valid_dataset)}")
    print_rank_0(f"Shape of input_data: {test_dataset[0].shape}")
    print_rank_0(f"Shape of label_data: {test_dataset[1].shape}\n")

    # DataLoaders creation:
    if not args.dist:
        train_sampler = RandomSampler(train_dataset)
        vaild_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        vaild_sampler = DistributedSampler(valid_dataset)

    train_dataloader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                num_workers=args.num_workers,
                                batch_size=args.per_device_train_batch_size)
    valid_dataloader = DataLoader(valid_dataset,
                                sampler=vaild_sampler,
                                num_workers=args.num_workers,
                                batch_size=args.per_device_valid_batch_size)
    
    return train_dataloader, valid_dataloader, test_dataset, data_scaler_list, dataset.x_site_matrix, dataset.y_site_matrix

class CFD_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the CFD data '''
    def __init__(self,
                 data_path,
                 coordinate_path,
                 data_select,
                 data_mean,
                 data_std,
                 data_max,
                 data_min,
                 data_range,
                 data_shape,
                 data_num,
                 data_previous,
                 data_after):

        # Read inputs
        self.data_type_num, self.height, self.width = data_shape
        self.data_previous = data_previous
        self.data_select = data_select
        self.data_after = data_after
        self.data_num = data_num
        self.data_range = data_range
        ny_range = self.data_range[0]
        nx_range = self.data_range[1]
        self.scaler_list = []
        self.custom_length = int(self.data_num - self.data_previous - self.data_after + 1)
        # Data Standard
        ''' #StandardScaler()
        for i in range(self.data_type_num):
            scaler = StandardScaler()
            scaler.mean_= data_mean[i]
            scaler.scale_ = data_std[i]
            self.scaler_list.append(scaler)
        '''
        for i in range(self.data_type_num):
            scaler = MinMaxScaler()
            custom_min = 0.0
            custom_max = 1.0
            scale = (custom_max - custom_min) / (data_max[i] - data_min[i])
            scaler.scale_ = scale
            scaler.min_= custom_min - data_min[i] * scale
            self.scaler_list.append(scaler)
        
        site_matrix = np.load(coordinate_path)
        data_matrix = np.load(data_path)
        self.data_total_num = data_matrix.shape[0]
        self.x_site_matrix = site_matrix[0][ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]
        self.y_site_matrix = site_matrix[1][ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]

        # Reshape data: (num, data_type,  height, width) -> (num, data_type, height * width)
        data_matrix = data_matrix.reshape(self.data_total_num, self.data_type_num,-1)
        for i in range(self.data_type_num):
            data_matrix[:,i] = self.scaler_list[i].transform(data_matrix[:,i])
        
        self.data_matrix = data_matrix.reshape(self.data_total_num, self.data_type_num, self.height, self.width)
        self.data_matrix = self.data_matrix[:, self.data_select, ny_range[0]:ny_range[1], nx_range[0]:nx_range[1]]

    def __getitem__(self, index):
        # Returns one sample at a time
        #batch: num of data_previous + num of data_after
        batch_num = self.data_previous + self.data_after 

        # in_la_data: (batch, data_type, height , width)
        in_la_data = self.data_matrix[index:index+batch_num]
        
        # Convert Data into PyTorch tensors
        in_la_data = torch.Tensor(in_la_data)            #(batch, data_type, height, width)

        inputs = in_la_data[:self.data_previous]          #(data_previous, data_type, height, width)
        labels = in_la_data[self.data_previous:]          #(data_after, data_type, height, width)
        return inputs, labels

    def __len__(self):
        # Returns the size of the dataset
        return self.custom_length
