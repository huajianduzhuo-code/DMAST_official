#!/bin/bash
# Setup script for the dmast conda environment

set -e  # Exit on error

echo "Starting dmast environment setup..."

# Create conda environment
echo "Creating conda environment 'dmast' with Python 3.11..."
conda create -n dmast python=3.11 -y

# Activate environment
echo "Activating dmast environment..."
eval "$(conda shell.bash hook)"
conda activate dmast

# Add auto-activation to bashrc if not already present
if ! grep -q "conda activate dmast" ~/.bashrc; then
    echo "conda activate dmast" >> ~/.bashrc
    echo "Added 'conda activate dmast' to ~/.bashrc"
fi

# Install compilation tools
echo "Installing compilation tools..."
pip install -U setuptools_scm ninja setuptools pip wheel packaging uv

# Install PyTorch 2.9.0 with CUDA 12.8
echo "Installing PyTorch 2.9.0 with CUDA 12.8..."
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install xformers first (exact version required by vLLM)
echo "Installing xformers 0.0.33+5d4b92a5.d20251029 from PyTorch wheels..."
pip install xformers==0.0.33+5d4b92a5.d20251029 --index-url https://download.pytorch.org/whl/cu128

# Install requirements from requirements.txt (vllm and flash-attn will be installed later)
echo "Installing remaining requirements from requirements.txt..."
MAX_JOBS=128 pip install --no-build-isolation -r requirements.txt # up to here

# Install flash-attn from source (to ensure compatibility with PyTorch 2.9.0)
echo "Installing flash-attn 2.7.2.post1 from source (this may take 5-10 minutes)..."
# Remove pre-built wheel if it exists
pip uninstall flash-attn -y
pip cache remove flash_attn
# Install from source with proper CUDA architecture flags
MAX_JOBS=128 pip install flash-attn==2.7.2.post1 --no-build-isolation

# Install vLLM first WITHOUT dependency checking (to avoid re-checking xformers)
echo "Installing custom vLLM (haoyu_custom branch) without strict dependency checking..."
MAX_JOBS=128 pip install --no-build-isolation git+https://github.com/huajianduzhuo-code/vllm.git@haoyu_custom#egg=vllm

# Disable bullseye-backports entries before installing Playwright dependencies.
echo "Fixing Debian sources for Playwright..."
if [ -f /etc/apt/sources.list ]; then
    sudo sed -i 's/deb https:\/\/deb.debian.org\/debian bullseye-backports main/# deb https:\/\/deb.debian.org\/debian bullseye-backports main/' /etc/apt/sources.list
    sudo sed -i 's/deb-src https:\/\/deb.debian.org\/debian bullseye-backports main/# deb-src https:\/\/deb.debian.org\/debian bullseye-backports main/' /etc/apt/sources.list
    echo "Commented out bullseye-backports in /etc/apt/sources.list"
else
    echo "/etc/apt/sources.list not found, skipping..."
fi

# Install Playwright
echo "Installing Playwright..."
playwright install
playwright install-deps
playwright install chromium

# Verify installation
echo ""
echo "Installation complete! Verifying..."
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import flash_attn; print(f'Flash-Attention: {flash_attn.__version__}')"
python -c "import flashinfer; print('FlashInfer: OK')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import playwright; print(f'Playwright: OK')"

echo ""
echo "dmast environment setup completed successfully!"
echo "To activate the environment, run: conda activate dmast"
