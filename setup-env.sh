#!/bin/bash

# Detect OS (Amazon Linux, Ubuntu, or Debian)
OS_NAME=$(grep -Ei '^(ID)=' /etc/os-release | cut -d'=' -f2 | tr -d '"')

echo "Detected OS: $OS_NAME"

# Function to install NVIDIA Drivers
install_nvidia_drivers() {
    echo "Installing NVIDIA Drivers..."
    if [[ "$OS_NAME" == "amzn" ]]; then
        sudo yum install -y nvidia-driver-latest-dkms
    elif [[ "$OS_NAME" == "ubuntu" || "$OS_NAME" == "debian" ]]; then
        sudo apt update -y
        sudo apt install -y nvidia-driver-535 nvidia-utils-535
    else
        echo "Unsupported OS for NVIDIA driver installation"
    fi
    echo "Verifying GPU..."
    nvidia-smi
}

# Function to install Python 3.12
install_python312() {
    echo "Installing Python 3.12..."
    if [[ "$OS_NAME" == "amzn" ]]; then
        sudo amazon-linux-extras enable python3.12
        sudo yum install -y python3.12
    elif [[ "$OS_NAME" == "ubuntu" || "$OS_NAME" == "debian" ]]; then
        sudo apt update -y
        sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip
    else
        echo "Unsupported OS for Python installation"
    fi
}

# Function to set Python 3.12 as default
set_python_default() {
    echo "Setting Python 3.12 as default..."
    if [[ "$OS_NAME" == "amzn" ]]; then
        sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
        sudo alternatives --config python3 <<EOF
1
EOF
    elif [[ "$OS_NAME" == "ubuntu" || "$OS_NAME" == "debian" ]]; then
        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
        sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
        sudo update-alternatives --config python3  # Choose Python 3.12 manually when prompted
    fi

    python3 --version
}

# Function to install Pip and JAX with CUDA 12 support
install_pip_and_jax() {
    echo "Installing pip and JAX..."
    if [[ "$OS_NAME" == "ubuntu" || "$OS_NAME" == "debian" ]]; then
        sudo apt install -y python3-pip
    fi
    
    python3 -m pip install --upgrade pip
    export PATH=$HOME/.local/bin:$PATH
    echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    
    pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

# Function to install project dependencies
install_requirements() {
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    else
        echo "Warning: requirements.txt not found!"
    fi
}

# Run all functions
install_nvidia_drivers
install_python312
set_python_default
install_pip_and_jax
install_requirements

# Configure memory usage for JAX
echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6' >> ~/.bashrc
source ~/.bashrc

# Verify installations
which pip3
pip3 --version
nvidia-smi
python3 --version

echo "Setup completed successfully!"