# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch.distributed as dist
import deepspeed
from utils.utils import print_rank_0, json2Parser, get_all_reduce_mean, save_json
from DataDefine import get_datloader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchsummary import summary
from rich.progress import track
matplotlib.use('AGG')
import time

class modeltrain():

    def __init__(self,model_data):

        (self.args,
         self.ds_config,
         self.net,
         self.optimizer,
         self.criterion
        ) = model_data
        pass

    def summary_model(self,input_size,device):
        if self.args.global_rank == 0:
            model = self.net.to(device)
            summary(model, input_size,0,device)

    def train_CFD(self,model_path):
        r"""Train the DeepCK net. The Loss will be saved as .png.

        Parameters
        ----------
        model_path : str
            The path of the model.

        """
        #Make Dataset
        train_dataloader, valid_dataloader, test_dataset, train_sampler, data_scaler_list, self.x_site_matrix, self.y_site_matrix \
            = get_datloader(self.args)
        #Training
        print_rank_0(f"lr = {self.args.learnrate}")
        data_record,model = self.train_model(train_dataloader,train_sampler, valid_dataloader,model_path)
        
        if self.args.global_rank == 0:
            self.plot_learning_curve(data_record, model_path, 300,title='CFD-ConvLSTM model')
            self.test_model(model, model_path,test_dataset,data_scaler_list)
        if dist.is_initialized():
            dist.barrier()


    def train_model(self, train_dataloader,train_sampler, valid_dataloader, model_path):
        ''' Model training '''
        model = self.net
        criterion = self.criterion
        n_epochs = self.args.max_epoch
        optimizer = self.optimizer
        args = self.args
        ds_config = self.ds_config
        # For recording training loss
        data_record = {
                "train_loss" : [],
                "valid_loss" : []
            }
        best_loss = 10.

        # Scheduler can be customized
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=, epochs=n_epochs)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        # Deepspeed initialize
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            config = ds_config,
            model=model, 
            optimizer = optimizer, 
            model_parameters=parameters,
            lr_scheduler = scheduler
            )
        rank = args.global_rank
        
        for epoch in range(0,n_epochs):
            if dist.is_initialized():
                train_sampler.set_epoch(epoch)
            tic = time.time()
            # ---------- Training ----------
            train_loss = self.train_epoch(model,train_dataloader,criterion,optimizer,lr_scheduler)
            #print_rank_0(optimizer.state_dict()['param_groups'][0]['lr'])
            data_record["train_loss"].append(train_loss) 
            print_rank_0(f'[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5e}')
            
            # ---------- Validation ----------
            valid_loss = self.validate(model,valid_dataloader,criterion)
            data_record["valid_loss"].append(valid_loss) 
            print_rank_0(f'[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5e}')
           
            # ---------- Comput time ----------
            data_time = time.time() - tic
            print_rank_0(f"data_time = {data_time:.4f}s\n")
            
            # ---------- Check Loss ----------
            if valid_loss < best_loss:
                best_loss = valid_loss
                if rank == 0:
                    checkpoint_path = os.path.join(model_path, 'checkpoint',
                                    f'model_{n_epochs}.pt')
                    save_dict = model.state_dict()
                    torch.save(save_dict, checkpoint_path)
                    print_rank_0('[{:03d}/{:03d}] saving model with loss {:.5e}'.format(epoch + 1, n_epochs, best_loss))
                if dist.is_initialized():
                    dist.barrier()
        
        json_file_path = os.path.join(model_path, 'checkpoint',
                                f'data_record.json')
        
        save_json(data_record,json_file_path)

        print_rank_0('Finished training after {} epochs'.format(epoch))
        return data_record,model
    
    def train_epoch(self,model,trainloader,criterion,optimizer,lr_scheduler):
        model.train()
        train_loss = []
        device = model.device
        #for i, (inputs, labels) in enumerate(trainloader):
        for inputs, labels in track(trainloader):
            # print_rank_0(f"shape of input: {inputs.shape}")
            # print_rank_0(f"shape of labels: {labels.shape}")
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            model.backward(loss)
            optimizer.step()
            if dist.is_initialized():
                loss = get_all_reduce_mean(loss)
            train_loss.append(loss.detach().cpu().item())
        lr_scheduler.step(np.mean(train_loss))
        return np.mean(train_loss)


    def validate(self,model,validloader,criterion):
        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        device = model.device
        # Iterate the validation set by batches.
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(validloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                pred = model(inputs)
                loss = criterion(pred, labels)
                valid_loss.append(loss.detach().cpu().item())
        return np.mean(valid_loss)

    def test_model(self, model, model_path,test_dataset,data_scaler_list):
        n_epochs = self.args.max_epoch
        checkpoint_path = os.path.join(model_path, 'checkpoint',
                                f'model_{n_epochs}.pt')
        model.load_state_dict(torch.load(checkpoint_path))
        print_rank_0(f"Successful load checkpoint!")
        data_type_num = self.args.data_type_num
        model.eval()
        device = model.device
        with torch.no_grad():
            inputs, labels = test_dataset
            inputs = inputs.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            pred = model(inputs)

        labels = labels.squeeze(0).cpu().numpy()
        pred = pred.squeeze(0).cpu().numpy()
        mode = "Computed"
        self.comput_test(labels, pred, mode,model_path)

        for i in range(data_type_num):
            labels[i] = data_scaler_list[i].inverse_transform(labels[i])
            pred[i] = data_scaler_list[i].inverse_transform(pred[i])
        mode = "Original"
        self.comput_test(labels, pred, mode,model_path)

    def comput_test(self, labels, pred,mode,model_path):
        data_type_num = self.args.data_type_num
        mse = [0 for _ in range(data_type_num)]
        rmse = [0 for _ in range(data_type_num)]
        mae = [0 for _ in range(data_type_num)]
        mre = [0 for _ in range(data_type_num)]

        for i in range(data_type_num):
            mse[i] = mean_squared_error(labels[i], pred[i])
            rmse[i] = np.sqrt(mse[i])
            mae[i] = mean_absolute_error(labels[i], pred[i])
            mre[i] = np.mean(np.abs(labels[i] - pred[i]) / (np.abs(labels[i]) + 1e-7))

            print_rank_0(f"{mode} {i} MSE: {mse[i]}")
            print_rank_0(f"{mode} {i} RMSE: {rmse[i]}")
            print_rank_0(f"{mode} {i} MAE: {mae[i]}")
            print_rank_0(f"{mode} {i} mre: {mre[i]}")

        self.plot_test(labels,pred,300,model_path,mode)

    def plot_test(self, labels,pred,dpi,model_path,mode):
        data_type_num = self.args.data_type_num
        pic_folder = os.path.join(model_path, 'pic',mode)
        os.makedirs(pic_folder, exist_ok=True)
        for i in range(data_type_num):
            min_max = [labels[i].min(), labels[i].max()]
            self.plot_test_figure(self, min_max, labels[i], i, "label", mode, pic_folder, dpi=dpi)
            self.plot_test_figure(self, min_max, pred[i], i, "pred", mode, pic_folder, dpi=dpi)

            min_max = [(labels[i]-pred[i]).min(), (labels[i]-pred[i]).max()]
            self.plot_test_figure(self, min_max, labels[i]-pred[i], i, "delt", mode, pic_folder, dpi=dpi)
            plt.close()

    def plot_test_figure(self, min_max, data, data_type, data_name, mode, pic_folder, dpi=300):
        cmap = 'RdBu_r'
        levels = np.linspace(min_max[0], min_max[1], 600)
        map = plt.contourf(self.x_site_matrix, self.y_site_matrix, data,levels,cmap=cmap) 
        pic_name = f'{data_type}_{mode}_{data_name}.png'
        ax = plt.gca()
        ax.set_aspect(1) 
        plt.colorbar(map,fraction=0.02, pad=0.03,
                     ticks=np.linspace(min_max[0], min_max[1], 5),
                     format = '%.1e')
        plt.title(f"{mode} {data_name} data of type {data_type}")
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        pic_path = os.path.join(pic_folder, pic_name)
        plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
        print_rank_0(f'{data_name} picture saved in {pic_path}')
        plt.close()

    def plot_learning_curve(self,loss_record, model_path, dpi=300, title=''):
        ''' Plot learning curve of your DNN (train & valid loss) '''
        total_steps = len(loss_record['train_loss'])
        x_1 = range(total_steps)
        x_2 = x_1[::len(loss_record['train_loss']) // len(loss_record['valid_loss'])]
        figure(figsize=(6, 4))
        plt.semilogx(x_2, loss_record['valid_loss'], c='tab:cyan', label='valid')
        plt.semilogx(x_1, loss_record['train_loss'], c='tab:red', label='train')
        plt.xlabel('Training steps')
        plt.ylabel('MSE loss')
        plt.title('Learning curve of {}'.format(title))
        plt.legend()

        pic_name = f'data_record.png'
        pic_folder = os.path.join(model_path, 'pic')
        os.makedirs(pic_folder, exist_ok=True)
        pic_path = os.path.join(pic_folder, pic_name)
        print_rank_0(f'simulation picture saved in {pic_path}')
        plt.savefig(pic_path, dpi=dpi)
        plt.close()


