from typing import Dict, List, Union
import numpy as np
import torch
from core import metric
import deepspeed
from utils import gather_tensors_batch,  ProgressBar


class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, model_data):
        super(Base_method, self).__init__()

        (self.args,
         self.ds_config,
         self.model,
         self.optimizer,
         self.scheduler,
         self.base_criterion
        ) = model_data

        self.dist = self.args.dist
        self.device = self.args.device
        self.rank = self.args.rank
        self.world_size = self.args.world_size
        self.scaler_list = None

        # setup metrics
        self.metric_list = [metric.lower() for metric in self.args.metrics]


    def init_distributed(self):
        """Initialize DeepSpeed training"""
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            args=self.args,
            config = self.ds_config,
            model = self.model, 
            optimizer = self.optimizer, 
            model_parameters=parameters,
            lr_scheduler = self.scheduler
            )

    def train_one_epoch(self,  train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError
    
    def cal_loss(self, pred_y, batch_y, **kwargs):
        """criterion of the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     scaler_list = self.scaler_list,
                                     metrics=self.metric_list, return_log=False)
                eval_res['loss'] = self.cal_loss(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length
        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     scaler_list = self.scaler_list,
                                     metrics=self.metric_list, return_log=False)
                eval_res['loss'] = self.cal_loss(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        return results_all

    def vali_one_epoch(self, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self.dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self.nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)

        eval_log = ""
        for k, v in results.items():
            v = v.mean()
            if k != "loss":
                eval_str = f"{k}: {v.mean():.5e}" if len(eval_log) == 0 else f", {k}: {v.mean():.5e}"
                eval_log += eval_str

        return results, eval_log

    def test_one_epoch(self, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self.dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self.nondist_forward_collect(test_loader, gather_data=True)

        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. 
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr