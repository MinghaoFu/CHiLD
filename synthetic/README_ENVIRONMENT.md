
## Quick Setup

### Option 1: Using Conda (Recommended)

1. Install Anaconda or Miniconda if not already installed.
2. Run the setup script (creates or updates the `child` environment from `environment.yml`):
   ```bash
   ./setup_environment.sh
   ```
3. Activate the environment:
   ```bash
   conda activate child
   ```

### Option 2: Manual Conda Setup

```bash
CONDA_ENVS_PATH=/data2/minghao/CHILD/.conda_envs conda env create -f environment.yml --solver=classic
conda activate child
```

If you prefer to avoid `environment.yml`, you can create the environment manually:

```bash
CONDA_ENVS_PATH=/data2/minghao/CHILD/.conda_envs conda create -y -n child python=3.10 pip --solver=classic
conda activate child
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia --solver=classic
pip install -r requirements.txt
```

## Dependencies

The main dependencies include:
- Python 3.10
- PyTorch (>=2.1,<2.3) with CUDA 11.8 builds
- PyTorch Lightning (>=2.0.0)
- Weights & Biases (wandb >=0.16.0)
- disentanglement-lib (>=1.4)
- NumPy, SciPy, Matplotlib, seaborn, tqdm, Optuna
- OpenCV Python, PyMunk, h5py, PyYAML
- ipdb and other utilities listed in `requirements.txt` (Torch packages are installed via conda rather than pip)

## CUDA Support

The conda environment installs the PyTorch CUDA 11.8 build. Make sure you have compatible NVIDIA drivers installed if you want to use GPU acceleration.



## Troubleshooting

1. If you see solver errors (e.g., libmamba plugins disabled), force the classic solver with `--solver=classic` or `CONDA_SOLVER=classic`
2. For disentanglement-lib installation issues, you may need to install it separately with specific versions
3. Make sure you have enough disk space for the conda environment (~2-3GB)
