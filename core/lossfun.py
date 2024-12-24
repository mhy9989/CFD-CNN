from torch import nn
import torch
import torch.nn.functional as F
from utils import print_log

def CD2(y, h):
    yy = torch.zeros_like(y, device=y.device)
    a1 = 1.0 / (2.0 * h)

    yy[..., 1:-1] = a1 * (y[..., 2:] - y[..., :-2])

    yy[..., 0] = (-3.0 * y[..., 0] + 4.0 * y[..., 1] - y[..., 2]) / (2.0 * h)

    yy[..., -1] = (y[..., -3] - 4.0 * y[..., -2] + 3.0 * y[..., -1]) / (2.0 * h)
    return yy

def CD4(y, h):
    yy = torch.zeros_like(y, device=y.device)
    a1 = 8.0 / (12.0 * h)
    a2 = 1.0 / (12.0 * h)

    yy[..., 2:-2] = a1 * (y[..., 3:-1] - y[..., 1:-3]) - a2 * (y[..., 4:] - y[..., :-4])

    yy[..., 0] = (-3.0 * y[..., 0] + 4.0 * y[..., 1] - y[..., 2]) / (2.0 * h)
    yy[..., 1] = (-2.0 * y[..., 0] - 3.0 * y[..., 1] + 6.0 * y[..., 2] - y[..., 3]) / (6.0 * h)

    yy[..., -2] = (y[..., -4] - 6.0 * y[..., -3] + 3.0 * y[..., -2] + 2.0 * y[..., -1]) / (6.0 * h)
    yy[..., -1] = (y[..., -3] - 4.0 * y[..., -2] + 3.0 * y[..., -1]) / (2.0 * h)
    return yy

def CD6(y, h):
    yy = torch.zeros_like(y, device=y.device)
    a1 = 1.0 / (60.0 * h)
    a2 = -3.0 / (20.0 * h)
    a3 = 3.0 / (4.0 * h)
    b1 = 8.0 / (12.0 * h)
    b2 = 1.0 / (12.0 * h)

    yy[..., 3:-3] = a1 * (y[..., 6:] - y[..., :-6]) + \
                a2 * (y[..., 5:-1] - y[..., 1:-5]) + \
                a3 * (y[..., 4:-2] - y[..., 2:-4])

    yy[..., 0] = (-3.0 * y[..., 0] + 4.0 * y[..., 1] - y[..., 2]) / (2.0 * h)
    yy[..., 1] = (-2.0 * y[..., 0] - 3.0 * y[..., 1] + 6.0 * y[..., 2] - y[..., 3]) / (6.0 * h)
    yy[..., 2] = b1 * (y[..., 3] - y[..., 1]) - b2 * (y[..., 4] - y[..., 0])

    yy[..., -3] = b1 * (y[..., -2] - y[..., -4]) - b2 * (y[..., -1] - y[..., -5])
    yy[..., -2] = (y[..., -4] - 6.0 * y[..., -3] + 3.0 * y[..., -2] + 2.0 * y[..., -1]) / (6.0 * h)
    yy[..., -1] = (y[..., -3] - 4.0 * y[..., -2] + 3.0 * y[..., -1]) / (2.0 * h)
    
    return yy


def diff(y, h, dim, mode = "CD2"):
    if dim == "y":
        y = y.transpose(-1, -2)
    elif dim == "x":
        y = y
    else:
        raise ValueError(f"false dim of diff: {dim}")
    
    if mode == "CD2":
        dy_dx = CD2(y, h)
    elif mode == "CD4":
        dy_dx = CD4(y, h)
    elif mode == "CD6":
        dy_dx = CD6(y, h)
    else:
        raise ValueError(f"false mode of diff: {mode}")

    return dy_dx.transpose(-1, -2) if dim == "y" else dy_dx


def MAE(pred_y, batch_y):
    return nn.L1Loss()(pred_y, batch_y)


def MSE(pred_y, batch_y):
    return nn.MSELoss()(pred_y, batch_y)


def SmoothL1Loss(pred_y, batch_y):
    return nn.SmoothL1Loss()(pred_y, batch_y)


def CEL(pred_y, batch_y):
    return nn.CrossEntropyLoss()(pred_y, batch_y)


def Log(pred_y, batch_y):
    return nn.LogSoftmax()(pred_y, batch_y)


def diff_div_reg(pred_y, batch_y, tau=0.1, eps=1e-12):
    B, T, C = pred_y.shape[:3]
    if T <= 2:  return 0
    gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
    gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
    softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
    softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
    loss_gap = softmax_gap_p * \
        torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
    return loss_gap.mean()


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2,dist=False):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            raise ValueError("param weight_decay can not <=0")
        self.dist = dist
        self.check_dist(model)
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(self.model)
 
    def to(self,device):
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.check_dist(model)
        self.weight_list=self.get_weight(self.model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, self.p)
        return reg_loss
 
    def check_dist(self, model):
        if self.dist:
            self.model=model.module
        else:
            self.model=model

    def get_weight(self,model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss


def GS(pred_y, batch_y, jac, mode = "CD2"):
    Akx = jac[0][0]
    Aky = jac[0][1]
    Aix = jac[1][0]
    Aiy = jac[1][1]
    nx = batch_y.shape[-1]
    ny = batch_y.shape[-2]
    hx = 1.0 / (nx - 1)
    hy = 1.0 / (ny - 1)
    dk_pred = diff(pred_y, hx, "x", mode)
    di_pred = diff(pred_y, hy, "y", mode)
    dk_batch = diff(batch_y, hx, "x", mode)
    di_batch = diff(batch_y, hy, "y", mode)

    dx_pred = dk_pred * Akx + di_pred * Aix
    dy_pred = dk_pred * Aky + di_pred * Aiy
    dx_batch = dk_batch * Akx + di_batch * Aix
    dy_batch = dk_batch * Aky + di_batch * Aiy
    return (MSE(dx_pred, dx_batch) + MSE(dy_pred, dy_batch)) / 2.0


def GS0(pred_y, batch_y, mode = "CD2"):
    hx=1.0
    hy=1.0
    dx_pred = diff(pred_y, hx, "x", mode)
    dy_pred = diff(pred_y, hy, "y", mode)
    dx_batch = diff(batch_y, hx, "x", mode)
    dy_batch = diff(batch_y, hy, "y", mode)
    return (MSE(dx_pred, dx_batch) + MSE(dy_pred, dy_batch)) / 2.0
