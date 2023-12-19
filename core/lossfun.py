from torch import nn
import torch
import torch.nn.functional as F

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

