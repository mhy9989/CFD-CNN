# A Flow Field Prediction Program (based on [OpenSTL](https://github.com/chengtan9907/OpenSTL))

## Overview

<details open>
<summary>Code Structures</summary>


- `core/` core training plugins and metrics.
- `methods/` contains training methods for various prediction methods.
- `Model/` contains the control files for various prediction methods and is used to store the model and training results.
- `models/` contains the main network architectures of various video prediction methods.
- `modules/` contains network modules and layers.
- `runfiles/` contains various startup scripts.
- `tool/` contains the font files `tool/font/`, pre-processing file `tool/pre-data.py`and standardized file `tool/comput_norm.py`
- `utils/` contains a variety of utilities for model training, model modules, plots, parsers, etc.
- `DataDefine.py` is used to get the flow field dataset and make a dataloader.
- `modelbuild.py` is used to build and initialize the model.
- `modeltrain.py` is used to train, validate, test and inference about model.
- `main.py` is the main function that runs the program.
- `inference.py` is used for model inference.
- `test.py` is used for model test.

</details>

## Multiple Spatio_temporal Attention (MSTA) Network

The code for **Multiple Spatio_temporal Attention (MSTA)** Layer can be found in `MSTA/MSTA.py`

The detailed **MSTA** code (Spatial Encoder/Decoder & MSTA module) can be found in `models/simvp_model.py` and `modules/simvp_modules.py`

- **Overview architecture**

<p align="center" width="100%">
<img src=".\fig\overview_architecture.jpg" width="70%" />
</p>


- **MSTA Block**

<p align="center" width="100%">
<img src=".\fig\MSTA_Block.jpg" width="30%" />
</p>

- **MSTA Layer**

<p align="center" width="100%">
<img src=".\fig\MSTA_flow_chart.jpg" width="50%" />
</p>

- **Large Kernel Attention (LKA) & Multiple Fusion Attention (MFA)**

<p align="center" width="100%">
<img src=".\fig\MSTA.jpg" alt="LKA & MFA" width="70%" />
</p>
