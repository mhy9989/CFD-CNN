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

        self.data_path_list = []
        # Read all data
        for i in range(1, 1+data_num):
            flow_data_path = os.path.join(data_path , f"flowxy-{i:04d}.dat")
            self.data_path_list.append(flow_data_path)

    def read_normalize(norm,mean,std,data=[]):
        scaler = StandardScaler()
        scaler.mean_=norm[mean]
        scaler.scale_=norm[std]
        data = scaler.transform(data)
        # Convert data into PyTorch tensors
        data = torch.FloatTensor(data)
        return scaler, data

    def normalize(self,data):
        scaler = StandardScaler().fit(data)
        data_mean = scaler.mean_.tolist()
        data_std = scaler.scale_.tolist()
        data = scaler.transform(data)
        # Convert data into PyTorch tensors
        data = torch.FloatTensor(data)
        return scaler, data, data_mean, data_std

    def __getitem__(self, index):
        # Returns one sample at a time
        # One sample: (input_channels, num, width*height)
        ### Betch: (batch_size, input_channels, num, width*height)

        nx = self.width
        ny = self.height
        ny_range = self.data_range[0]
        nx_range = self.data_range[1]
        data_type_num = self.data_type_num
        batch_num = self.data_previous + self.data_after #batch: num of data_previous + num of data_after
        input_data_list = np.zeros((batch_num, data_type_num,ny*nx))

        # Read all data
        # input_data_list: (batch, data_type, height * width)
        for i in range(index,index+batch_num):
            data = np.loadtxt(self.data_path_list[i],skiprows=2)
            x_site = data[:nx*ny,0]
            y_site = data[:nx*ny,1]
            for j in range(data_type_num):
                input_data_list[i-index, j] = data[:nx*ny,j+2]
        
        # Reshape data: (ny * nx) -> (ny, nx)
        x_site_matrix = x_site.reshape(ny, nx)
        y_site_matrix = y_site.reshape(ny, nx)

        # data_matrix: (batch, data_type, height, width)
        data_matrix = np.zeros((batch_num, data_type_num, ny, nx))
        for j in range(data_type_num):
            input_data_list[:,j] = self.scaler_list[j].transform(input_data_list[:,j])
            for i in range(batch_num):
                matrix = input_data_list[i, j]
                # Reshape data: (ny * nx) -> (ny, nx)
                data_matrix[i, j] = matrix.reshape(ny, nx)
        
        # Convert Data into PyTorch tensors
        self.x_site_matrix = torch.Tensor(x_site_matrix) #(height, width)
        self.y_site_matrix = torch.Tensor(y_site_matrix) #(height, width)
        self.data_matrix = torch.Tensor(data_matrix)     #(batch, data_type, height, width)

        self.x_site_matrix = self.x_site_matrix[ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]
        self.y_site_matrix = self.y_site_matrix[ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]

        input = self.data_matrix[:self.data_previous]            #(data_previous, data_type, height, width)
        label = self.data_matrix[self.data_previous:batch_num]   #(data_after, data_type, height, width)
        input_out = input[:,self.data_select,ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]] #(data_after, data_select, height, width)
        label_out = label[:,self.data_select,ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]] #(data_after, data_select, height, width)


        return input_out, label_out

    def __len__(self):
        # Returns the size of the dataset
        return self.custom_length
