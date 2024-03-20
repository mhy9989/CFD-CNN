# A flow field prediction program modified based on [OpenSTL](https://github.com/chengtan9907/OpenSTL)

## Overview

<details open>
<summary>Code Structures</summary>


- `core/` core training plugins and metrics.
- `methods/` contains training methods for various prediction methods.
- `Model/` contains the control files for various prediction methods and is used to store the model and training results.
- `models`/ contains the main network architectures of various video prediction methods.
- `modules/` contains network modules and layers.
- `runfiles/` contains various startup scripts.
- `tools/` contains the font files `tool/font/`, pre-processing file `tool/pre-data.py`and standardized file `tool/comput_norm.py`
- `utils/` contains a variety of utilities for model training, model modules, plots, parsers, etc.
- `DataDefine.py` is used to get the flow field dataset and make a dataloader.
- `modelbuild.py ` is used to build and initialize the model.
- `modeltrain.py ` is used to train, validate, test and inference about model.
- `main.py` is the main function that runs the program.
- `inference.py` is used for model inference.
- `test.py` is used for model test.

</details>

## Multi Spatio_temporal Attention (MSTA)

The code for **Multi Spatio_temporal Attention (MSTA)** Layer can  be found in `MSTA/MSTA.py`

The detailed **MSTA** code (Spatial Encoder/Decoder & MSTA module) can be found `models/simvp_model.py` and `modules/simvp_modules.py`

- **Overview architecture**

<img src="E:\git\CFD-CNN\fig\overview_architecture.png" style="zoom:10%;" />

- **MSTA Block**

<img src=".\fig\MSTA_Block.png" style="zoom: 10%;" />

- **MSTA Layer**

<img src=".\fig\MSTA_flow_chart.png" style="zoom:10%;" />

- **Large Kernel Attention (LKA) & Multi-dimensional Channel Attention (MCA)**

<img src=".\fig\MSTA.png" alt="LKA & MCA" style="zoom: 10%;" />
