import time
import torch
from timm.utils import AverageMeter
from models import PredRNN_Model
from utils import (reduce_tensor, reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling, get_progress)
from .base_method import Base_method


class PredRNN(Base_method):
    r"""PredRNN

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://dl.acm.org/doi/abs/10.5555/3294771.3294855>`_.

    """

    def __init__(self, args, ds_config, base_criterion):
        Base_method.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(self.args)

    def build_model(self, args):
        num_hidden = self.args.num_hidden
        return PredRNN_Model(num_hidden, args).to(self.device)
    
    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y)
        return loss

    def predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        # reverse schedule sampling
        if self.args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.args.data_previous
        _, img_channel, img_height, img_width = self.args.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        test_dat = reshape_patch(test_ims, self.args.patch_size)
        test_ims = test_ims[:, :, :, :, :img_channel]

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - mask_input - 1,
            img_height // self.args.patch_size,
            img_width // self.args.patch_size,
            self.args.patch_size ** 2 * img_channel)).to(self.device)
            
        if self.args.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.args.data_previous - 1, :, :] = 1.0

        img_gen = self.model(test_dat, real_input_flag)
        img_gen = reshape_patch_back(img_gen, self.args.patch_size)
        pred_y = img_gen[:, -self.args.data_after:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_y

    def train_one_epoch(self, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        log_buffer = "Training..."
        progress = get_progress()
        
        end = time.time()
        with progress:
            if self.rank == 0:
                train_pbar = progress.add_task(description=log_buffer, total=len(train_loader))
            
            for batch_x, batch_y in train_loader:
                data_time_m.update(time.time() - end)
                if self.by_epoch or not self.dist:
                    self.optimizer.zero_grad()

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # preprocess
                ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
                ims = reshape_patch(ims, self.args.patch_size)
                if self.args.reverse_scheduled_sampling == 1:
                    real_input_flag = reserve_schedule_sampling_exp(
                        num_updates, ims.shape[0], self.args)
                else:
                    eta, real_input_flag = schedule_sampling(
                        eta, num_updates, ims.shape[0], self.args)

                img_gen = self.model(ims, real_input_flag)
                loss = self.cal_loss(img_gen, ims[:, 1:])

                if not self.dist:
                    loss.backward()
                else:
                    self.model.backward(loss)

                if self.by_epoch or not self.dist:
                    self.optimizer.step()
                else:
                    self.model.step()
                
                if not self.dist:
                    self.scheduler.step()

                torch.cuda.synchronize()
                num_updates += 1

                if self.dist:
                    losses_m.update(reduce_tensor(loss), batch_x.size(0))
                else:
                    losses_m.update(loss.item(), batch_x.size(0))

                if self.rank == 0:
                    log_buffer = 'train loss: {:.4e}'.format(loss.item())
                    log_buffer += ' | data time: {:.4e}'.format(data_time_m.avg)
                    progress.update(train_pbar, advance=1)#, description=f"{log_buffer}")

                end = time.time()  # end for

        if self.by_epoch:
            self.scheduler.step(epoch)

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return num_updates, losses_m, eta

