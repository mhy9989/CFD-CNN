from easydict import EasyDict as edict

def default_parser():
    default_values = {
    "setup_config":{
        "seed": 2023,
        "diff_seed": False,
        "per_device_train_batch_size": 1,
        "per_device_valid_batch_size": 1,
        "num_workers": 1,
        "method": "SimVP",
        "max_epoch": 100,
        "lossfun": "MSE",
        "load_from": False,
        "if_continue": True,
        "regularization": 0.0,
        "if_display_method_info": False,
        "mem_log": True,
        "empty_cache": True,
        "metrics":["MSE", "RMSE", "MAE", "MRE", "SSIM", "MAX_RE"],
        "fps": True
    },
    "data_config": {
        "data_path": "./CFD_data.npy",
        "mesh_path": "./CFD_mesh.npy",
        "data_num": 1001,
        "data_type": ["u", "v", "T", "P"],
        "data_select": [0, 1, 2, 3],
        "data_width": 1181,
        "data_height": 220,
        "data_range": [
        [0, 208],
        [0, 1168]
        ],
        "data_mean": [
        0.7160895679671404, 
        0.10879711642281557, 
        3.15980087314138,
        6.678367111021137
        ],
        "data_std": [
        0.34781700305722757, 
        0.17350806486928866, 
        2.2907320741672956,
        8.661780748427242
        ],
        "data_max": [1.4213101, 1.08254902, 12.95459985, 78.43348863],
        "data_min": [-0.55474735, -0.68369049, 0.77961313, 0.24014292],
        "data_scaler": "Standard",
        "data_previous": 5,
        "data_after": 1,
        "valid_ratio": 0.1
    },
    "optim_config": {
        "optim": "Adam",
        "lr": 0.001,
        "filter_bias_and_bn": False,
        "log_step": 1,
        "opt_eps": "",
        "opt_betas": "",
        "momentum": 0.9,
        "weight_decay": 0,
        "early_stop_epoch": -1
    },
    "sched_config": {
        "sched": "onecycle",
        "min_lr": 1e-6,
        "warmup_lr": 1e-5,
        "warmup_epoch": 0,
        "decay_rate": 0.1,
        "decay_epoch": 100,
        "lr_k_decay": 1.0,
        "final_div_factor": 1e4
    },
    "model_config": {
        "model_type": "gSTA",
        "hid_S": 128,
        "hid_T": 1024,
        "N_S": 8,
        "N_T": 16,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3
    },
    "ds_config":{
        "offload": False,
        "zero_stage": 0,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0
    }
    }

    return edict(default_values)
