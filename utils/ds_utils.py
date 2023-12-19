# -*- coding: utf-8 -*-
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(args,steps_per_print):
    offload = args.offload
    stage = args.zero_stage
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
    }
    ds_config = args.ds_config
    default_ds_config = {
        "train_batch_size": args.per_device_train_batch_size * \
                            args.world_size * \
                            args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "steps_per_print": steps_per_print,
        "zero_optimization": zero_opt_dict,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    for key in default_ds_config.keys():
        if key not in ds_config.keys():
            ds_config[key] = default_ds_config[key]

    return ds_config

