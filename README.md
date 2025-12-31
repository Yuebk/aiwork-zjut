# CIFAR-10 CNN Projects

This workspace contains three PyTorch image classification models for CIFAR-10:

- `cifar10_lenet`: Baseline LeNet-style model
- `cifar10_lenet_advanced`: LeNet with BatchNorm + data augmentation
- `cifar10_vgg_simple`: Simplified VGG-style CNN

Also includes utilities for parallel training and model visualization.

## Quick Setup (pip)

```powershell
# Create and activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

## Conda Setup (GPU-ready)

```powershell
# Create conda env (requires conda)
conda env create -f environment.yml
conda activate cifar10-cnn-env
```

Notes:
- `environment.yml` includes `pytorch-cuda=12.1` and channels `pytorch`, `nvidia`, `conda-forge` for GPU support.
- If you don't have an NVIDIA GPU or don’t want CUDA, remove the `pytorch-cuda=12.1` line before creating the env.

## Run Training

```powershell
# Train all models in parallel (heavy resource usage)
python train_all.py

# Train individual models
cd cifar10_lenet; python main.py
cd ../cifar10_lenet_advanced; python main.py
cd ../cifar10_vgg_simple; python main.py
```

Environment tuning for parallel training:
```powershell
# Reduce memory pressure on Windows
set BATCH_SIZE=32
set NUM_WORKERS_PER_PROC=0
python train_all.py
```

## Visualization

```powershell
# Export ONNX, TensorBoard graphs, Torchinfo and Torchviz
python visualize_models.py

# View TensorBoard
tensorboard --logdir=model_visualizations/tensorboard_logs
```

For Torchviz PNGs, Graphviz system binaries may be required. If only `.dot` files are generated:
```powershell
dot -Tpng model_visualizations\lenet_torchviz.dot -o model_visualizations\lenet_torchviz.png
```

## Data Analysis

Generate dataset visualizations (class distribution and channel stats):
```powershell
python analysis/data_analysis.py
```
Outputs are saved to `reports/img/`.
