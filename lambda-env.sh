#!/bin/bash

# Ensure the system is Ubuntu-based
OS_NAME=$(grep -Ei '^(ID)=' /etc/os-release | cut -d'=' -f2 | tr -d '"')

if [[ "$OS_NAME" != "ubuntu" ]]; then
    echo "This script is optimized for Ubuntu-based Lambda Cloud instances."
    exit 1
fi

echo "Detected OS: $OS_NAME"

# Function to update pip and install JAX with CUDA 12 support
install_pip_and_jax() {
    echo "Upgrading pip..."
    python3 -m ensurepip --default-pip
    python3 -m pip install --upgrade pip

    echo "Installing JAX for CUDA 12..."
    pip3 install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

# Function to install Tkinter and GUI dependencies
install_tkinter() {
    echo "Installing Tkinter and GUI dependencies..."
    if [[ "$OS_NAME" == "ubuntu" || "$OS_NAME" == "debian" ]]; then
        sudo apt update -y
        sudo apt install -y python3-tk python3.10-tk tk-dev
        sudo apt install -y libxrender1 libxext6 libsm6 libxft2
        sudo apt update -y
        sudo apt install -y python3-tk
    else
        echo "Unsupported OS for Tkinter installation"
    fi
}

# Function to install project dependencies
install_requirements() {
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip3 install -r requirements.txt
    else
        echo "Warning: requirements.txt not found!"
    fi
}

#Degrade numpy and pandas to match lambda instance
downgrade_numpy() {
    echo "Degrade numpy to less than version 2.xx"
    pip3 install --upgrade "numpy<2" --force-reinstall
    pip3 uninstall numpy pandas datasets
    pip3 install --force-reinstall numpy pandas datasets
    pip install "numpy<2" --force-reinstall
    pip install --upgrade --force-reinstall numpy jax jaxlib dm-haiku ml_dtypes
}

# Run functions
install_pip_and_jax
install_tkinter
install_requirements
downgrade_numpy

# Verify installations
pip3 --version
python3 --version

echo "Minimal setup completed successfully!"
