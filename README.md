# Multiple Spatio-temporal Attention (MSTA) Network

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

The code for **Multiple Spatio-temporal Attention (MSTA)** Layer can be found in `MSTA/MSTA.py`

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

## 📖 Citation

If you find this project helpful, please consider citing our paper:

```bibtex
@article{MEN2025109685,
  title   = {Multiple spatio-temporal attention network: A deep convolutional network for spatio-temporal evolution prediction of flow fields},
  journal = {Computer Physics Communications},
  volume  = {315},
  pages   = {109685},
  year    = {2025},
  issn    = {0010-4655},
  doi     = {https://doi.org/10.1016/j.cpc.2025.109685},
  url     = {https://www.sciencedirect.com/science/article/pii/S0010465525001870},
  author  = {Hongyuan Men and Ji Zhang and Yixuan Mao and Xinliang Li and Guoan Zhao and Hongwei Liu},
  keywords= {Turbulence, Flow field prediction, Convolutional network, Channel attention, Gradient sharpening}
}
