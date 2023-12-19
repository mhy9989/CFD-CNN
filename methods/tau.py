import time
import torch
from timm.utils import AverageMeter

from utils import reduce_tensor
from .simvp import SimVP
from rich.progress import track
from core.lossfun import diff_div_reg
from tqdm import tqdm


class TAU(SimVP):
    r"""TAU

    Implementation of `Temporal Attention Unit: Towards Efficient Spatiotemporal 
    Predictive Learning <https://arxiv.org/abs/2206.12126>`_.

    """

    def __init__(self, model_data):
        SimVP.__init__(self, model_data)

    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y) + self.args.alpha * diff_div_reg(pred_y, batch_y)
        return loss
    
    def train_one_epoch(self, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader
        #track(train_loader, description="Training...")
        end = time.time()
        for batch_x, batch_y in train_pbar:
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
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for
        
        if self.by_epoch:
            self.scheduler.step(epoch)

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return num_updates, losses_m, eta
