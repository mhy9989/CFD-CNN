import time
import torch
from timm.utils import AverageMeter

from utils import reduce_tensor, get_progress
from .base_method import Base_method
from rich.progress import track
from tqdm import tqdm


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, model_data):
        Base_method.__init__(self, model_data)
        self.by_epoch = self.args.by_epoch

    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y)
        return loss

    def predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.data_after == self.args.data_previous:
            pred_y = self.model(batch_x)
        elif self.args.data_after < self.args.data_previous:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.data_after]
        elif self.args.data_after > self.args.data_previous:
            pred_y = []
            d = self.args.data_after // self.args.data_previous
            m = self.args.data_after % self.args.data_previous
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
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
                if self.by_epoch:
                    self.optimizer.zero_grad()

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                pred_y = self.predict(batch_x)
                loss = self.cal_loss(pred_y, batch_y)

                if not self.dist:
                    losses_m.update(loss.item(), batch_x.size(0))

                self.model.backward(loss)
                if self.by_epoch:
                    self.optimizer.step()
                else:
                    self.model.step()

                torch.cuda.synchronize()
                num_updates += 1

                if self.dist:
                    losses_m.update(reduce_tensor(loss), batch_x.size(0))

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
