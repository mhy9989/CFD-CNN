import os.path as osp
import time
import numpy as np
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table
from DataDefine import get_datloader
import torch
from deepspeed.accelerator import get_accelerator
from core import metric, Recorder
from methods import method_maps
from utils import (plot_test_figure, check_dir, print_log, weights_to_cpu,
                   measure_throughput, output_namespace)


class modeltrain(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, model_data, model_path, mode = "train", test_num = -1):
        """Initialize experiments (non-dist as an example)"""
        self.model_data = model_data
        self.args = model_data[0]
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self.epoch = 0
        self.max_epochs = self.args.max_epoch
        self.steps_per_epoch = self.args.steps_per_epoch
        self.rank = self.args.rank
        self.world_size = self.args.world_size
        self.dist = self.args.dist
        self.early_stop = self.args.early_stop_epoch
        self.model_path = model_path
        self.best_loss = 100.
        self.test_num = test_num
        self.mode = mode

        self.preparation()
        print_log(output_namespace(self.args))
        if self.args.if_display_method_info:
            self.display_method_info()


    def preparation(self):
        """Preparation of basic experiment setups"""
        if self.early_stop <= self.max_epochs // 5:
            self.early_stop = self.max_epochs * 2

        self.checkpoints_path = osp.join(self.model_path, 'checkpoints')
        # build the method
        self.build_method()
        # load checkpoint
        if self.args.load_from and self.mode == "train":
            if self.args.load_from == True:
                self.args.load_from = 'latest'
            self.load(name=self.args.load_from)
        # prepare data
        self.get_data()


    def build_method(self):
        self.method = method_maps[self.args.method.lower()](self.model_data)
        self.method.model.eval()
        # setup ddp training
        if self.dist:
            self.method.init_distributed()


    def get_data(self):
        """Prepare datasets and dataloaders"""
        if self.mode == "train":
            (self.train_loader, 
            self.vali_loader, 
            self.test_loader, 
            self.scaler_list, 
            self.x_mesh, 
            self.y_mesh,
            self.jac) = get_datloader(self.args)
            self.method.jac = torch.Tensor(self.jac).to(self.device)
            self.method.scaler_list = self.scaler_list
            if self.vali_loader is None:
                self.vali_loader = self.test_loader
        else:
            custom_length = int(self.args.data_num - self.args.data_previous - self.args.data_after + 1)
            if abs(self.test_num) < (custom_length):
                (self.test_loader, 
                self.scaler_list, 
                self.x_mesh, 
                self.y_mesh) = get_datloader(self.args, "test", self.test_num)
                self.method.scaler_list = self.scaler_list
            else:
                print_log(f"num {self.test_num} out of data range")
                raise ValueError


    def save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self.epoch + 1,
            'optimizer': self.method.optimizer.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()
            }
        torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint
    

    def save_checkpoint(self, name=''):
        """Saving models data to checkpoints"""
        checkpoint = weights_to_cpu(self.method.model.state_dict()) \
                if not self.dist else weights_to_cpu(self.method.model.module.state_dict())
        torch.save(checkpoint, osp.join(self.checkpoints_path, f'{name}.pth'))
        del checkpoint


    def load(self, name=''):
        """Loading models from the checkpoint"""
        filename = osp.join(self.checkpoints_path, f'{name}.pth')
        try:
            checkpoint = torch.load(filename, map_location='cpu')
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self.load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self.epoch = checkpoint['epoch']
            self.method.optimizer.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            print_log(f"Successful optimizer state_dict, Lr: {cur_lr:.5e}")
        del checkpoint


    def load_from_state_dict(self, state_dict):
        if self.dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)
        print_log(f"Successful load model state_dict")

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        if self.args.method in ['simvp', 'tau']:
            input_dummy = torch.ones(1, self.args.data_previous, C, H, W).to(self.device)
        elif self.args.method == 'crevnet':
            # crevnet must use the batchsize rather than 1
            input_dummy = torch.ones(self.args.batch_size, 20, C, H, W).to(self.device)
        elif self.args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, self.args.data_previous, C, H, W).to(self.device)
            _tmp_input2 = torch.ones(1, self.args.data_after, C, H, W).to(self.device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(self.device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        # elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
        #     Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
        #     Cp = self.args.patch_size ** 2 * C
        #     _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
        #     _tmp_flag = torch.ones(1, self.args.data_after - 1, Hp, Wp, Cp).to(self.device)
        #     input_dummy = (_tmp_input, _tmp_flag)
        # elif self.args.method == 'predrnnv2':
        #     Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
        #     Cp = self.args.patch_size ** 2 * C
        #     _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
        #     _tmp_flag = torch.ones(1, self.args.total_length - 2, Hp, Wp, Cp).to(self.device)
        #     input_dummy = (_tmp_input, _tmp_flag)
        elif self.args.method == 'dmvfn':
            input_dummy = torch.ones(1, 3, C, H, W, requires_grad=True).to(self.device)
        elif self.args.method == 'prednet':
           input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)

    def train(self):
        """Training loops of methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self.max_epochs // 10, 30), 
                            rank = self.rank, dist=self.dist, max_epochs = self.max_epochs)
        num_updates = self.epoch * self.steps_per_epoch
        vali_loss = False
        early_stop = False
        eta = 1.0  # PredRNN variants
        for epoch in range(self.epoch, self.max_epochs):
            if self.args.empty_cache:
                torch.cuda.empty_cache()

            if self.dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, eta = self.method.train_one_epoch(self.train_loader,
                                                                      epoch, num_updates, eta)

            self.epoch = epoch
            if self.args.valid_ratio != 0:
                with torch.no_grad():
                    vali_loss = self.vali()
            
            cur_lr = self.method.current_lr()
            cur_lr = sum(cur_lr) / len(cur_lr)
            print_log('Epoch: {0}, Steps: {1} | Lr: {2:.5e} | Train Loss: {3:.5e} | Vali Loss: {4:.5e}'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss if vali_loss else 0))
            
            if self.args.mem_log:
                MemAllocated = round(get_accelerator().memory_allocated() / 1024**3, 2)
                MaxMemAllocated = round(get_accelerator().max_memory_allocated() / 1024**3, 2)
                print_log(f"MemAllocated: {MemAllocated} GB, MaxMemAllocated: {MaxMemAllocated} GB")

            early_stop = recorder(loss_mean.avg, vali_loss, self.method.model, self.model_path, epoch)
            self.best_loss = recorder.val_loss_min

            if self.rank == 0:
                self.save(name='latest')

            if epoch > self.early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))
                break
            print_log("")
            
        self.save_checkpoint("last_checkpoint")

        if not check_dir(self.model_path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        time.sleep(1)

    def vali(self):
        """A validation loop during training"""
        results, eval_log = self.method.vali_one_epoch(self.vali_loader)
        print_log('Val_metrics\t'+eval_log)

        return results['loss'].mean()

    def test(self):
        """A testing loop of methods"""
        best_model_path = osp.join(self.checkpoints_path, 'checkpoint.pth')
        self.load_from_state_dict(torch.load(best_model_path))
        
        results = self.method.test_one_epoch(self.test_loader)
        
        channel_names = self.args.data_use
        metric_list = self.method.metric_list

        eval_res_o=""
        # Computed
        eval_res_o, eval_log_o = metric(results['preds'], results['trues'],
                                    metrics=metric_list, channel_names=channel_names, mode = "Computed")
        results['metrics'] = np.array([eval_res_o['mae'], eval_res_o['mse'], eval_res_o['mre']])
        print_log(f"{eval_log_o}")

        if self.rank == 0:
            self.plot_test(results['inputs'], results['trues'], results['preds'], "Computed")
            folder_path = osp.join(self.model_path, 'saved', "Computed")
            check_dir(folder_path)

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        eval_res=""
        # Original
        if self.scaler_list:
            results_n = self.de_norm(results)
            eval_res, eval_log = metric(results['preds'], results['trues'],
                                        self.scaler_list,
                                        metrics=metric_list, channel_names=channel_names, mode = "Original")
            results['metrics'] = np.array([eval_res['mae'], eval_res['mse'], eval_res['mre']])
            print_log(f"{eval_log}")

            if self.rank == 0:
                self.plot_test(results_n['inputs'], results_n['trues'], results_n['preds'], "Original")
                folder_path = osp.join(self.model_path, 'saved', "Original")
                check_dir(folder_path)

                for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), results_n[np_data])

        return eval_res_o, eval_res

    def inference(self):
        """A inference loop of methods"""
        best_model_path = osp.join(self.model_path, 'checkpoint.pth')
        self.load_from_state_dict(torch.load(best_model_path))

        results = self.method.test_one_epoch(self, self.test_loader)

        if self.rank == 0:
            folder_path = osp.join(self.model_path, 'saved')
            check_dir(folder_path)
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        return None
    
    def plot_test(self, inputs, tures, preds, mode, dpi = 300, dir_name = "pic"):
        B, T, C, H, W = tures.shape
        if B != 1 or T != 1:
            raise ValueError("B and T must is 1")
        inputs = inputs[0, -1]
        tures = tures.reshape(C, H, W)
        preds = preds.reshape(C, H, W)
        data_select_num = self.args.data_select_num
        pic_folder = osp.join(self.model_path, dir_name, mode)
        check_dir(pic_folder)
        for i in range(data_select_num):
            min_max = [inputs[i].min(), inputs[i].max()]
            plot_test_figure(self.x_mesh, self.y_mesh, min_max, inputs[i], 
                             self.args.data_use[i], "inputs", mode, pic_folder, dpi)
            min_max = [tures[i].min(), tures[i].max()]
            plot_test_figure(self.x_mesh, self.y_mesh, min_max, tures[i], 
                             self.args.data_use[i], "label", mode, pic_folder, dpi)
            plot_test_figure(self.x_mesh, self.y_mesh, min_max, preds[i], 
                             self.args.data_use[i], "pred", mode, pic_folder, dpi)
            min_max = [(tures[i]-preds[i]).min(), (tures[i]-preds[i]).max()]
            plot_test_figure(self.x_mesh, self.y_mesh, min_max, tures[i]-preds[i], 
                             self.args.data_use[i], "delt", mode, pic_folder, dpi)
        return None
    
    def de_norm(self, results):
        results_ori = {}
        for name in results.keys():
            if name in ['inputs', 'trues', 'preds'] and self.scaler_list:
                B, T, C, H, W = results[name].shape
                results_ori[name] = np.zeros((B, T, C, H, W))
                for b in range(B):
                    for t in range(T):
                        for c in range(C):
                            results_ori[name][b, t, c] = self.scaler_list[c].inverse_transform(results[name][b, t, c])
            else:
                results_ori[name] = results[name]
        return results_ori
