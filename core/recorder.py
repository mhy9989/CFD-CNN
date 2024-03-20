import numpy as np
import torch
from utils import print_log, weights_to_cpu, save_json, plot_learning_curve
import os.path as osp


class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=30, rank=0, dist=True, 
                 max_epochs = 0, method = "CFD-Conv"):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = early_stop_time
        self.rank = rank
        self.dist = dist
        self.max_epochs = max_epochs
        self.loss_recoder =  {
                "train_loss" : [],
                "valid_loss" : []
            }
        self.method = method

    def __call__(self, train_loss, val_loss, model, path , epoch):
        self.loss_recoder["train_loss"].append(float(train_loss))
        if val_loss:
            self.loss_recoder["valid_loss"].append(float(val_loss)) 
        val_loss = val_loss if val_loss else train_loss
        score = -val_loss 
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.decrease_time = 0
        else:
            self.decrease_time += 1

        if (epoch +1)% 50 == 0 or (epoch +1) == self.max_epochs and self.rank == 0:
            json_file_path = osp.join(path,"checkpoints","loss_recoder.json")
            save_json(self.loss_recoder, json_file_path)
            print_log(f'Save the loss_recoder of {epoch+1} epochs')
            plot_learning_curve(self.loss_recoder, path, 300, title=f'{self.method} model')

        return self.decrease_time > self.early_stop_time 
    
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print_log(f'Loss decreased ({self.val_loss_min:.5e} --> {val_loss:.5e}).  Saving best model ...')
        if self.rank == 0:
            torch.save(weights_to_cpu(model.state_dict()) \
                if not self.dist else weights_to_cpu(model.module.state_dict()), osp.join(path, "checkpoints", "checkpoint.pth"))
        self.val_loss_min = val_loss

