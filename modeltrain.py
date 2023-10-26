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
        self.rank = self.args.global_rank
        pass

    def summary_model(self,input_size,device):
        if self.rank == 0:
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
        train_dataloader, valid_dataloader, test_dataset, data_scaler_list, self.x_site_matrix, self.y_site_matrix \
            = get_datloader(self.args)
        #Training
        data_record,model = self.train_model(train_dataloader, valid_dataloader,model_path)
        
        if self.rank == 0:
            self.plot_learning_curve(data_record, model_path, 300,title='CFD-Conv model')
            self.test_model(model, model_path,test_dataset,data_scaler_list)
        if self.args.dist:
            dist.barrier()

    def test_CFD(self,model_path,num):
        r"""test the DeepCK net. The Loss will be saved as .png.

        Parameters
        ----------
        model_path : str
            The path of the model.

        """
        #Make Dataset
        custom_length = int(self.data_num - self.data_previous - self.data_after + 1)
        if int(num) < (custom_length) and num >= 0:
            test_dataset, data_scaler_list, self.x_site_matrix, self.y_site_matrix \
                = get_datloader(self.args, "test", num)
        else:
            print_rank_0(f"num {num} out of data range")
        model=self.net
        #Training
        if self.rank == 0:
            self.test_model(model, model_path,test_dataset,data_scaler_list,dir_name = f"pic_{num}")
        if self.args.dist:
            dist.barrier()


    def train_model(self, train_dataloader, valid_dataloader, model_path):
        ''' Model training '''
        n_epochs = self.args.max_epoch
        # For recording training loss
        data_record = {
                "train_loss" : [],
                "valid_loss" : []
            }
        best_loss = 10.
        best_epoch = 0

        # Scheduler can be customized
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=self.args.steps_per_epoch, epochs=n_epochs)
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        # Deepspeed initialize
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=self.args,
            config = self.ds_config,
            model=self.net, 
            optimizer = self.optimizer, 
            model_parameters=parameters,
            lr_scheduler = self.scheduler
            )
        print_rank_0(f"device of model is: {model.device}")
        for epoch in range(0,n_epochs):
            if self.args.dist:
                train_dataloader.sampler.set_epoch(epoch)
            tic = time.time()
            # ---------- Training ----------
            train_loss = self.train_epoch(model,train_dataloader,self.criterion)
            #print_rank_0(optimizer.state_dict()['param_groups'][0]['lr'])
            data_record["train_loss"].append(train_loss) 
            print_rank_0(f'[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5e}')
            
            # ---------- Validation ----------
            if self.args.valid_ratio != 0:
                valid_loss = self.validate(model,valid_dataloader,self.criterion)
                data_record["valid_loss"].append(valid_loss) 
                print_rank_0(f'[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5e}')
           
            # ---------- Comput time ----------
            data_time = time.time() - tic
            print_rank_0(f"data_time = {data_time:.4f}s")
            
            # ---------- Check Loss ----------
            if self.args.valid_ratio != 0:
                new_loss = valid_loss
            else:
                new_loss = train_loss
            if new_loss < best_loss:
                best_loss = new_loss 
                best_epoch = epoch + 1
                if self.rank == 0:
                    checkpoint_path = os.path.join(model_path, 'checkpoint',
                                    f'model_{n_epochs}.pt')
                    save_dict = model.state_dict()
                    torch.save(save_dict, checkpoint_path)
                    print_rank_0('[{:03d}/{:03d}] saving model with loss {:.5e}'.format(epoch + 1, n_epochs, best_loss))
                if dist.is_initialized():
                    dist.barrier()
            
            if (epoch +1)% 50 == 0 or (epoch +1) == n_epochs:
                json_file_path = os.path.join(model_path, 'checkpoint',
                                f'data_record.json')
                save_json(data_record,json_file_path)
                print_rank_0('Save the  data_record of {} epochs'.format(epoch+1))
            
            print_rank_0(f"\n")

        print_rank_0('Finished training after {} epochs'.format(epoch+1))
        print_rank_0(f'Best loss is: {best_loss:.5e}')
        print_rank_0(f'Best loss epoch is: {best_epoch:03d}')
        print_rank_0(f'\n')
        return data_record,model
    
    def _predict(self, inputs, model):
        """Forward the model"""
        if self.args.data_after == self.args.data_previous:
            pred_y = model(inputs)
        elif self.args.data_after < self.args.data_previous:
            pred_y = model(inputs)
            pred_y = pred_y[:, :self.args.data_after]
        elif self.args.data_after > self.args.data_previous:
            pred_y = []
            d = self.args.data_after // self.args.data_previous
            m = self.args.data_after % self.args.data_previous
            
            cur_seq = inputs.clone()
            for _ in range(d):
                cur_seq = model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_epoch(self,model,trainloader,criterion):
        """Train the model with train_loader."""
        model.train()
        train_loss = []
        device = model.device
        train_pbar = track(trainloader, description="Training...") if self.rank == 0 and not self.args.dist else trainloader
        for inputs, labels in train_pbar:
            # print_rank_0(f"shape of input: {inputs.shape}")
            # print_rank_0(f"shape of labels: {labels.shape}")
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = self._predict(inputs, model)
            loss = criterion(pred, labels)
            model.backward(loss)
            model.step()
            if self.args.dist:
                loss = get_all_reduce_mean(loss)
            train_loss.append(loss.detach().cpu().item())
        return np.mean(train_loss)


    def validate(self,model,validloader,criterion):
        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        device = model.device
        # Iterate the validation set by batches.
        valid_pbar = track(validloader, description="Validing...") if self.rank == 0 and not self.args.dist else validloader
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                pred = model(inputs)
                loss = criterion(pred, labels)
                valid_loss.append(loss.detach().cpu().item())
        return np.mean(valid_loss)

    def test_model(self, model, model_path,test_dataset,data_scaler_list,dir_name = "pic"):
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
        labels = labels[0,0].cpu().numpy()
        pred = pred[0,0].cpu().numpy()
        mode = "Computed"
        self.comput_test(labels, pred, mode,model_path, dir_name)

        for i in range(data_type_num):
            labels[i] = data_scaler_list[i].inverse_transform(labels[i])
            pred[i] = data_scaler_list[i].inverse_transform(pred[i])
        mode = "Original"
        self.comput_test(labels, pred, mode,model_path, dir_name)

    def comput_test(self, labels, pred,mode,model_path, dir_name = "pic"):
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

        self.plot_test(labels,pred,300,model_path,mode, dir_name)
        print_rank_0(f"\n")

    def plot_test(self, labels,pred,dpi,model_path,mode, dir_name = "pic"):
        data_type_num = self.args.data_type_num
        pic_folder = os.path.join(model_path, dir_name, mode)
        os.makedirs(pic_folder, exist_ok=True)
        for i in range(data_type_num):
            min_max = [labels[i].min(), labels[i].max()]
            self.plot_test_figure(min_max, labels[i], i, "label", mode, pic_folder, dpi)
            self.plot_test_figure(min_max, pred[i], i, "pred", mode, pic_folder, dpi)

            min_max = [(labels[i]-pred[i]).min(), (labels[i]-pred[i]).max()]
            self.plot_test_figure(min_max, labels[i]-pred[i], i, "delt", mode, pic_folder, dpi)

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

    def plot_learning_curve(self,loss_record, model_path,dpi=300, title='', dir_name = "pic"):
        ''' Plot learning curve of your DNN (train & valid loss) '''
        total_steps = len(loss_record['train_loss'])
        x_1 = range(total_steps)
        if self.args.valid_ratio != 0:
            x_2 = x_1[::len(loss_record['train_loss']) // len(loss_record['valid_loss'])]
            plt.semilogy(x_2, loss_record['valid_loss'], c='tab:cyan', label='valid')
        plt.semilogy(x_1, loss_record['train_loss'], c='tab:red', label='train')
        plt.xlabel('Training steps')
        plt.ylabel('MSE loss')
        plt.title('Learning curve of {}'.format(title))
        plt.legend()

        pic_name = f'data_record.png'
        pic_folder = os.path.join(model_path, dir_name)
        os.makedirs(pic_folder, exist_ok=True)
        pic_path = os.path.join(pic_folder, pic_name)
        print_rank_0(f'simulation picture saved in {pic_path}')
        plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
        plt.close()


