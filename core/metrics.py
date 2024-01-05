import numpy as np
from skimage.metrics import structural_similarity as cal_ssim


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true):
    return np.mean(np.abs(pred-true))


def MRE(pred, true):
    return np.mean(np.abs(pred-true) / (np.abs(true) + 1e-20))


def MSE(pred, true):
    return np.mean((pred-true)**2)


def RMSE(pred, true):
    return np.sqrt(np.mean((pred-true)**2))


def MXRE(pred, true):
    return np.max(np.abs(pred-true) / (np.abs(true) + 1e-20))


def SSIM(pred, true):
    try:
        B, T, C, H, W = true.shape
    except:
        B, T, H, W = true.shape
        C = 1
    pred = pred.reshape(B, T, C, H, W)
    true = true.reshape(B, T, C, H, W)
    ssim = []
    for b in range(B):
        for t in range(T):
            for c in range(C):
                ssim.append(cal_ssim(pred[b, t, c], true[b, t, c], data_range = true[b, t, c].max() - true[b, t, c].min()))
    return np.mean(ssim)


def metric(pred, true, scaler_list = None, metrics=['mae', 'mse', 'mre'],
           clip_range=[0, 1], channel_names=None, return_log=True, mode = None):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        true (tensor): The prediction values of output prediction.
        metric (str | list[str]): Metrics to be evaluated.
        clip_range (list): Range of prediction to prevent overflow.
        channel_names (list | None): The name of different channels.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    if scaler_list :
        B, T, C, H, W = true.shape
        for b in range(B):
            for t in range(T):
                for c in range(C):
                    pred[b, t, c] = scaler_list[c].inverse_transform(pred[b, t, c])
                    true[b, t, c] = scaler_list[c].inverse_transform(true[b, t, c])
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'mre', 'mxre']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert len(channel_names) > 0
        c_group = len(channel_names)
    else:
        channel_names, c_group = None, None
    
    eval_res = {}
    if channel_names is None:
        eval_res_type = 0
    else:
        eval_res_type = 1

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true)
        else:
            mse_sum = 0.
            eval_res['mse'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['mse'][f'mse_{str(c_name)}'] = MSE(pred[:, :, i], true[:, :, i])
                mse_sum += eval_res['mse'][f'mse_{str(c_name)}']
            eval_res['mse']['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true)
        else:
            mae_sum = 0.
            eval_res['mae'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['mae'][f'mae_{str(c_name)}'] = MAE(pred[:, :, i], true[:, :, i])
                mae_sum += eval_res['mae'][f'mae_{str(c_name)}']
            eval_res['mae']['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true)
        else:
            rmse_sum = 0.
            eval_res['rmse'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['rmse'][f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i], true[:, :, i])
                rmse_sum += eval_res['rmse'][f'rmse_{str(c_name)}']
            eval_res['rmse'][f'rmse'] = rmse_sum / c_group
    
    if 'mre' in metrics:
        if channel_names is None:
            eval_res['mre'] = MRE(pred, true)
        else:
            mre_sum = 0.
            eval_res['mre'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['mre'][f'mre_{str(c_name)}'] = MRE(pred[:, :, i], true[:, :, i])
                mre_sum += eval_res['mre'][f'mre_{str(c_name)}']
            eval_res['mre']['mre'] = mre_sum / c_group
    
    if 'mxre' in metrics:
        if channel_names is None:
            eval_res['mxre'] = MXRE(pred, true)
        else:
            mxre_sum = 0.
            eval_res['mxre'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['mxre'][f'mxre_{str(c_name)}'] = MXRE(pred[:, :, i], true[:, :, i])
                mxre_sum += eval_res['mxre'][f'mxre_{str(c_name)}']
            eval_res['mxre']['mxre'] = mxre_sum / c_group

    # pred = np.maximum(pred, clip_range[0])
    # pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        if channel_names is None:
            eval_res['ssim'] = SSIM(pred, true)
        else:
            ssim_sum = 0.
            eval_res['ssim'] = {}
            for i, c_name in enumerate(channel_names):
                eval_res['ssim'][f'ssim_{str(c_name)}'] = SSIM(pred[:, :, i], true[:, :, i])
                ssim_sum += eval_res['ssim'][f'ssim_{str(c_name)}']
            eval_res['ssim']['ssim'] = ssim_sum / c_group

    if return_log:
        if eval_res_type == 0:
            for k, v in eval_res.items():
                eval_str = f"{k}:{v:.5e}" if len(eval_log) == 0 else f", {k}:{v:.5e}"
                eval_log += eval_str
        elif eval_res_type == 1:
            eval_str_tt = ""
            for k, v in eval_res.items():
                eval_log = ""
                for vk, vv in v.items():
                    eval_str = f"{vk:<6}:{vv:>12.5e}" if len(eval_log) == 0 else f", {vk:<6}:{vv:>12.5e}"
                    eval_log += eval_str
                eval_str_tt += eval_log + "\n"
            eval_log = eval_str_tt

        if mode:
            eval_log = f"{mode}:\n" + eval_log

    return eval_res, eval_log
