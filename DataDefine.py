# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import print_log, jac
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split, Subset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_datloader(args, mode = "train", infer_num = [-1], infer_step = 1):
    """Generate dataloader"""
    dataset = CFD_Dataset(args, infer_step)
    data_scaler_list = [dataset.scaler_list[i] for i in args.data_select] if dataset.scaler_list else None
    print_log(f"Length of all dataset: {len(dataset)}")
   
    if mode == "inference":
        inference_list = []
        batch_num = args.data_previous + infer_step * args.data_after 
        for i in infer_num:
            if i < 0:
                ii = args.data_num + i
            else:
                ii = i
            if (ii + batch_num) > (args.data_num):
                print_log(f"Error inference num: {ii}")
                raise EOFError
            inference_list.append(ii)
        inference_data = dataset.inference_data(inference_list) # B, T, C, H, W
        inference_dataset = infer_Dataset(inference_data[:, :args.data_previous],
                                          inference_data[:, args.data_previous : args.data_after+args.data_previous])
        infer_loader = DataLoader(inference_dataset,
                                num_workers=args.num_workers,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False)
        return inference_list, inference_data, infer_loader, data_scaler_list, dataset.x_mesh, dataset.y_mesh
    
    # Split dataset into training dataset, validation dataset and test_dataset
    indices = list(range(len(dataset)))
    indices_train_valid = indices[:-args.text_num]
    indices_test = indices[-args.text_num:]

    train_valid_dataset = Subset(dataset, indices_train_valid)
    test_dataset = Subset(dataset, indices_test)
    
    trainlen = int((1 - args.valid_ratio) * len(train_valid_dataset))
    lengths = [trainlen, len(train_valid_dataset) - trainlen]
    train_dataset, valid_dataset = random_split(train_valid_dataset, lengths)

    print_log(f"Length of input dataset: {len(dataset)}")
    print_log(f"Length of train_dataset: {len(train_dataset)}")
    print_log(f"Length of valid_dataset: {len(valid_dataset)}")
    print_log(f"Length of test_dataset: {len(valid_dataset)}")
    print_log(f"Shape of input_data: {test_dataset[0][0].shape}")
    print_log(f"Shape of label_data: {test_dataset[0][1].shape}")


    # DataLoaders creation:
    if not args.dist:
        train_sampler = RandomSampler(train_dataset)
        vaild_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        vaild_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                batch_size=args.per_device_train_batch_size)
    vali_loader = DataLoader(valid_dataset,
                                sampler=vaild_sampler,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size)
    test_loader = DataLoader(test_dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False)
    return train_loader, vali_loader, test_loader, data_scaler_list, dataset.x_mesh, dataset.y_mesh, dataset.jac


class CFD_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the CFD data '''
    def __init__(self, args, infer_step = 1):
        # Read inputs
        self.data_type_num, self.height, self.width = args.data_shape
        self.data_previous = args.data_previous
        self.data_select = args.data_select
        self.data_after = args.data_after
        self.data_num = args.data_num
        self.data_range = args.data_range
        ny_range = self.data_range[0]
        nx_range = self.data_range[1]
        self.infer_step = infer_step  # only inference used
        self.scaler_list = []
        self.custom_length = int(self.data_num - self.data_previous - self.data_after + 1)
        # Data Standard
        if args.data_scaler == "Standard":
            for i in range(self.data_type_num):
                scaler = StandardScaler()
                scaler.mean_= args.data_mean[i]
                scaler.scale_ = args.data_std[i]
                self.scaler_list.append(scaler)
        elif args.data_scaler == "MinMax":
            for i in range(self.data_type_num):
                scaler = MinMaxScaler()
                custom_min = 0.0
                custom_max = 1.0
                scale = (custom_max - custom_min) / (args.data_max[i] - args.data_min[i])
                scaler.scale_ = scale
                scaler.min_= custom_min - args.data_min[i] * scale
                self.scaler_list.append(scaler)
        elif not args.data_scaler:
            self.scaler_list = None
        else:
            print_log(f"Error data_scaler type: {args.data_scaler}")
            raise EOFError
        
        mesh = np.load(args.mesh_path)
        data_matrix = np.load(args.data_path)[:self.data_num]

        self.x_mesh = mesh[0][ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]
        self.y_mesh = mesh[1][ny_range[0]:ny_range[1],nx_range[0]:nx_range[1]]

        # Reshape data: (num, data_type,  height, width) -> (num, data_type, height * width)
        data_matrix = data_matrix.reshape(self.data_num, self.data_type_num,-1)
        if self.scaler_list:
            for i in range(self.data_type_num):
                data_matrix[:,i] = self.scaler_list[i].transform(data_matrix[:,i])
        
        self.data_matrix = data_matrix.reshape(self.data_num, self.data_type_num, self.height, self.width)
        self.data_matrix = self.data_matrix[:, self.data_select, ny_range[0]:ny_range[1], nx_range[0]:nx_range[1]]
        
        # comput jac
        self.jac = jac(self.x_mesh, self.y_mesh)

        del mesh
        del data_matrix


    def inference_data(self, inference_list):
        inference_data = []
        batch_num = self.data_previous + self.infer_step * self.data_after 

        for i in inference_list:
            inference_data.append(self.data_matrix[i:i+batch_num])
        
        return np.array(inference_data) # B, T, C, H, W
    

    def __getitem__(self, index):
        # Returns one sample at a time
        #batch: num of data_previous + num of data_after
        batch_num = self.data_previous + self.data_after 

        # in_la_data: (batch, data_type, height , width)
        in_la_data = self.data_matrix[index:index+batch_num]

        # Convert Data into PyTorch tensors
        in_la_data = torch.Tensor(in_la_data)             #(batch, data_type, height, width)
        inputs = in_la_data[:self.data_previous]          #(data_previous, data_type, height, width)
        labels = in_la_data[self.data_previous:]          #(data_after, data_type, height, width)
        return inputs, labels


    def __len__(self):
        # Returns the size of the dataset
        return self.custom_length
    

class infer_Dataset(Dataset):
    ''' Dataset for muti_inference the CFD data '''
    def __init__(self, inputs, labels):
        # Read inputs
        self.inputs = torch.Tensor(inputs) # B, T, C, H, W
        self.labels = torch.Tensor(labels) # B, T, C, H, W
        self.custom_length = len(inputs)


    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


    def __len__(self):
        # Returns the size of the dataset
        return self.custom_length