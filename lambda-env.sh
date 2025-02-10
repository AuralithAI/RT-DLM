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

# Function to install project dependencies
install_requirements() {
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip3 install -r requirements.txt
    else
        echo "Warning: requirements.txt not found!"
    fi
}

# Run functions
install_pip_and_jax
install_requirements

# Verify installations
pip3 --version
python3 --version

echo "Minimal setup completed successfully!"
