import time
import torch
from timm.utils import AverageMeter

from utils import reduce_tensor, get_progress
from .simvp import SimVP
from models import SimVP_Model
from core.lossfun import diff_div_reg

class TAU(SimVP):
    r"""TAU

    Implementation of `Temporal Attention Unit: Towards Efficient Spatiotemporal 
    Predictive Learning <https://arxiv.org/abs/2206.12126>`_.

    """

    def __init__(self, args, ds_config, base_criterion):
        SimVP.__init__(self, args, ds_config, base_criterion)
        self.model = self.build_model(args)

    def build_model(self, args):
        return SimVP_Model(**args).to(self.device)

    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model."""
        loss = self.base_criterion(pred_y, batch_y) + self.args.alpha * diff_div_reg(pred_y, batch_y)
        return loss
    
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

                pred_y = self.predict(batch_x)
                loss = self.cal_loss(pred_y, batch_y)

                if not self.dist:
                    loss.backward()
                else:
                    self.model.backward(loss)

                if self.by_epoch or not self.dist:
                    self.optimizer.step()
                else:
                    self.model.step()

                if not self.dist and not self.by_epoch:
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
