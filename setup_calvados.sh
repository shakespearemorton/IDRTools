#!/bin/bash
# Exit immediately if a command fails
set -e

echo "==> Pinning Python to version 3.11..."
uv python pin 3.11

echo "==> Creating virtual environment..."
uv venv

echo "==> Installing PyTorch for CUDA 12.6..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing OpenMM and matching CUDA plugin (preventing ABI mismatch)..."
uv pip install openmm==8.2.0 openmm-cuda-12==8.2.0 mdanalysis==2.9.0 mdtraj==1.11

echo "==> Installing Metapredict and CALVADOS..."
uv pip install metapredict git+https://github.com/KULL-Centre/CALVADOS.git

echo "==> Setup complete!"
