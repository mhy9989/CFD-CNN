import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from models import MAU_Model
from utils import reduce_tensor, schedule_sampling, get_progress
from .base_method import Base_method


class MAU(Base_method):
    r"""MAU

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, args, ds_config, base_criterion):
        Base_method.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(self.args)

    def build_model(self, args):
        num_hidden = self.args.num_hidden
        return MAU_Model(num_hidden, args).to(self.device)

    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y)
        return loss

    def predict(self, batch_x, batch_y, **kwargs):
        """Forward the model."""
        _, img_channel, img_height, img_width = self.args.in_shape

        # preprocess
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - self.args.data_previous - 1,
            img_height // self.args.patch_size,
            img_width // self.args.patch_size,
            self.args.patch_size ** 2 * img_channel)).to(self.device)

        img_gen = self.model(test_ims, real_input_flag)
        pred_y = img_gen[:, -self.args.data_after:, :]

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
                eta, real_input_flag = schedule_sampling(eta, num_updates, ims.shape[0], self.args)

                img_gen = self.model(ims, real_input_flag)
                loss = self.cal_loss(img_gen, ims.permute(0, 1, 4, 2, 3).contiguous()[:, 1:])

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
