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

def get_datloader(args, mode = "train", infer_num = 0):
    """Generate dataloader"""
    dataset = CFD_Dataset(args)
    data_scaler_list = [dataset.scaler_list[i] for i in args.data_select] if dataset.scaler_list else None
    print_log(f"Length of all dataset: {len(dataset)}")
    # Split dataset into training dataset, validation dataset and test_dataset
    # The last line of data is test data
    if mode == "inference":
        if infer_num < 0:
            infer_num += len(dataset)
        '''
        test_dataset = Subset(dataset, [test_num])
        test_loader = DataLoader(test_dataset,
                                num_workers=0,
                                batch_size=len(test_dataset))
        return test_loader, data_scaler_list, dataset.x_mesh, dataset.y_mesh
        '''
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
                                batch_size=args.per_device_valid_batch_size)
    return train_loader, vali_loader, test_loader, data_scaler_list, dataset.x_mesh, dataset.y_mesh, dataset.jac


class CFD_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the CFD data '''
    def __init__(self,args):
        # Read inputs
        self.data_type_num, self.height, self.width = args.data_shape
        self.data_previous = args.data_previous
        self.data_select = args.data_select
        self.data_after = args.data_after
        self.data_num = args.data_num
        self.data_range = args.data_range
        ny_range = self.data_range[0]
        nx_range = self.data_range[1]
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