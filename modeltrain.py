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
from skimage.metrics import structural_similarity
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
        self.train_model(train_dataloader, valid_dataloader,model_path)
        
        if self.rank == 0:
            self.test_model(model_path,test_dataset,data_scaler_list)
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
        custom_length = int(self.args.data_num - self.args.data_previous - self.args.data_after + 1)
        if int(num) < (custom_length):
            test_dataset, data_scaler_list, self.x_site_matrix, self.y_site_matrix \
                = get_datloader(self.args, "test", num)
        else:
            print_rank_0(f"num {num} out of data range")
            return
        #Training
        if self.rank == 0:
            self.test_model(model_path,test_dataset,data_scaler_list,dir_name = f"pic_{num}")
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
        early_stop_cnt = 0

        # Scheduler can be customized
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, steps_per_epoch=self.args.steps_per_epoch, epochs=n_epochs)
        parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        model = self.net
        if self.args.lf_load:
            checkpoint_path = os.path.join(model_path, 'checkpoint',
                                f'model_{n_epochs}.pt')
            state_dict = torch.load(checkpoint_path, map_location=self.args.device)
            model.load_state_dict(state_dict)
            print(f"Successful load model state_dict")
        # Deepspeed initialize
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=self.args,
            config = self.ds_config,
            model=model, 
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
            print_rank_0(f"lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
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
                early_stop_cnt = 0
                if self.rank == 0:
                    checkpoint_path = os.path.join(model_path, 'checkpoint',
                                    f'model_{n_epochs}.pt')
                    save_dict = model.module.state_dict()
                    torch.save(save_dict, checkpoint_path)
                    print_rank_0(f'[{epoch + 1:03d}/{n_epochs:03d}] Saving model with loss {best_loss:.5e}')
                if dist.is_initialized():
                    dist.barrier()
            else:
                early_stop_cnt += 1
            
            if (epoch +1)% 50 == 0 or (epoch +1) == n_epochs:
                json_file_path = os.path.join(model_path, 'checkpoint',
                                f'data_record.json')
                save_json(data_record,json_file_path)
                print_rank_0(f'Save the data_record of {epoch+1} epochs')
                if self.rank == 0:
                    self.plot_learning_curve(data_record, model_path, 300,title='CFD-Conv model')
            
            print_rank_0(f"\n")
            if early_stop_cnt > self.args.early_stop:
                break

        print_rank_0(f'Finished training after {epoch+1} epochs')
        print_rank_0(f'Best loss is: {best_loss:.5e}')
        print_rank_0(f'Best loss epoch is: {best_epoch}')
        print_rank_0(f'\n')
        return 
    
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
                pred = self._predict(inputs, model)
                loss = criterion(pred, labels)
                if self.args.dist:
                    loss = get_all_reduce_mean(loss)
                valid_loss.append(loss.detach().cpu().item())
        return np.mean(valid_loss)

    def test_model(self, model_path,test_dataset,data_scaler_list,dir_name = "pic"):
        n_epochs = self.args.max_epoch
        device = self.args.device
        model = self.net
        checkpoint_path = os.path.join(model_path, 'checkpoint',
                                f'model_{n_epochs}.pt')
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successful load model state_dict")
        
        model.to(self.args.device)
        data_select_num = self.args.data_select_num
        model.eval()
        with torch.no_grad():
            inputs, labels = test_dataset
            inputs = inputs.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            pred = model(inputs)
        labels = labels[0,0].cpu().numpy()
        pred = pred[0,0].cpu().numpy()
        mode = "Computed"
        self.comput_test(labels, pred, mode,model_path, dir_name)

        for i in range(data_select_num):
            labels[i] = data_scaler_list[self.args.data_select[i]].inverse_transform(labels[i])
            pred[i] = data_scaler_list[self.args.data_select[i]].inverse_transform(pred[i])
        mode = "Original"
        self.comput_test(labels, pred, mode, model_path, dir_name)

    def comput_test(self, labels, pred,mode, model_path, dir_name = "pic"):
        data_select_num = self.args.data_select_num
        mse = [0 for _ in range(data_select_num)]
        rmse = [0 for _ in range(data_select_num)]
        mae = [0 for _ in range(data_select_num)]
        mre = [0 for _ in range(data_select_num)]
        ssim = [0 for _ in range(data_select_num)]
        max_re = [0 for _ in range(data_select_num)]
        

        for i in range(data_select_num):
            mse[i] = mean_squared_error(labels[i], pred[i])
            rmse[i] = np.sqrt(mse[i])
            mae[i] = mean_absolute_error(labels[i], pred[i])
            mre[i] = np.mean(np.abs(labels[i] - pred[i]) / (np.abs(labels[i]) + 1e-10))
            max_re[i] = np.max(np.abs(labels[i] - pred[i]) / (np.abs(labels[i]) + 1e-10))
            ssim[i] = structural_similarity(labels[i],pred[i],data_range=labels[i].max() - labels[i].min())

            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} MSE: {mse[i]:.5e}")
            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} RMSE: {rmse[i]:.5e}")
            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} MAE: {mae[i]:.5e}")
            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} MRE: {mre[i]:.5e}")
            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} SSIM: {ssim[i]:.5e}")
            print_rank_0(f"{mode} {self.args.data_type[self.args.data_select[i]]} MAX_RE: {max_re[i]:.5e}")

        self.plot_test(labels,pred,300,model_path,mode, dir_name)
        print_rank_0(f"\n")

    def plot_test(self, labels,pred,dpi,model_path,mode, dir_name = "pic"):
        data_select_num = self.args.data_select_num
        pic_folder = os.path.join(model_path, dir_name, mode)
        os.makedirs(pic_folder, exist_ok=True)
        for i in range(data_select_num):
            min_max = [labels[i].min(), labels[i].max()]
            self.plot_test_figure(min_max, labels[i], self.args.data_type[self.args.data_select[i]], "label", mode, pic_folder, dpi)
            self.plot_test_figure(min_max, pred[i], self.args.data_type[self.args.data_select[i]], "pred", mode, pic_folder, dpi)

            min_max = [(labels[i]-pred[i]).min(), (labels[i]-pred[i]).max()]
            self.plot_test_figure(min_max, labels[i]-pred[i], self.args.data_type[self.args.data_select[i]], "delt", mode, pic_folder, dpi)

    def plot_test_figure(self, min_max, data, data_select, data_name, mode, pic_folder, dpi=300):
        cmap = 'RdBu_r'
        levels = np.linspace(min_max[0], min_max[1], 600)
        map = plt.contourf(self.x_site_matrix, self.y_site_matrix, data,levels,cmap=cmap) 
        pic_name = f'{data_select}_{mode}_{data_name}.png'
        ax = plt.gca()
        ax.set_aspect(1) 
        plt.colorbar(map,fraction=0.02, pad=0.03,
                     ticks=np.linspace(min_max[0], min_max[1], 5),
                     format = '%.1e')
        plt.title(f"{mode} {data_name} data of type {data_select}")
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
        print_rank_0(f'Simulation picture saved in {pic_path}')
        plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
        plt.close()


