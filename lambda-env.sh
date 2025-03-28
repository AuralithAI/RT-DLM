#!/bin/bash

# Ensure the system is Ubuntu-based
OS_NAME=$(grep -Ei '^(ID)=' /etc/os-release | cut -d'=' -f2 | tr -d '"')

if [[ "$OS_NAME" != "ubuntu" ]]; then
    echo "Error: This script is designed for Ubuntu-based Lambda Cloud instances."
    exit 1
fi

echo "Detected OS: $OS_NAME"

# Variables
VENV_DIR="$HOME/venv"
PYTHON="python3"
PIP="$VENV_DIR/bin/pip"
AWS_CLI="$VENV_DIR/bin/aws"

# Function to set up virtual environment
setup_venv() {
    echo "Setting up virtual environment in $VENV_DIR..."
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated."
}

# Function to update pip
update_pip() {
    echo "Upgrading pip..."
    $PIP install --upgrade pip
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

# Function to install dependencies from requirements.txt
install_requirements() {
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        $PIP install -r requirements.txt
    else
        echo "Error: requirements.txt not found in the current directory!"
        exit 1
    fi
}

# Function to install JAX with CUDA 12 support
install_jax_cuda() {
    echo "Installing JAX with CUDA 12 support..."
    $PIP install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
}

# Function to install AWS CLI
install_aws_cli() {
    echo "Installing AWS CLI..."
    $PIP install --upgrade awscli
}

# Main execution
echo "Starting minimal setup..."

# Update system packages (minimal set for Python and GPU support)
sudo apt update -y
sudo apt install -y python3 python3-pip python3-venv

# Set up virtual environment and activate it
setup_venv

# Update pip and install dependencies
update_pip
install_requirements
install_jax_cuda
install_aws_cli

# Verify installations
echo "Verifying installations..."
$VENV_DIR/bin/python --version
$PIP --version
$VENV_DIR/bin/python -c "import jax; print('JAX version:', jax.__version__)"
$AWS_CLI --version

echo "Minimal setup completed successfully!"
echo "To activate the virtual environment, run: source $VENV_DIR/bin/activate"