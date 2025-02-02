#!/bin/bash

# Update instance
sudo yum update -y

# Install NVIDIA drivers
sudo yum install -y nvidia-driver-latest-dkms

# Check GPU
nvidia-smi

# Enable and install Python 3.12 (For Amazon Linux 2)
sudo amazon-linux-extras enable python3.12
sudo yum install -y python3.12

# Set Python 3.12 as default
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo alternatives --config python3 <<EOF
1
EOF

# Verify Python version
python3 --version

# Install pip after setting Python 3.12 as default
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

# Update PATH for user-installed binaries
export PATH=$HOME/.local/bin:$PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify pip installation
which pip3
pip3 --version

# Install JAX with CUDA 12 support
pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install project dependencies (only if requirements.txt exists)
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi