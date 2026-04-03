# Installation

## Quick setup (recommended)

```bash
git clone https://github.com/avarth/UrbanSolarCarver.git
cd UrbanSolarCarver
python setup_env.py
```

This creates a virtual environment, auto-detects your NVIDIA GPU, installs the matching PyTorch wheels, and installs USC in editable mode.

```bash
python setup_env.py --cpu      # force CPU-only (no CUDA)
python setup_env.py --dry-run  # show what would be installed
```

## Manual installation

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev]
```

## Requirements

- Python >= 3.9
- NVIDIA GPU + CUDA driver (recommended; CPU mode available but significantly slower)

## Verifying the installation

```python
import torch
print(torch.cuda.is_available())  # True if GPU is available

from urbansolarcarver import load_config
print("USC ready")
```
