#!/usr/bin/env python3
"""
RT-DLM AGI Dependencies Installation Script
Ensures all dependencies are correctly installed with compatible versions
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print status"""
    print(f"[INFO] {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False

def main():
    """Main installation process"""
    print("RT-DLM AGI Dependencies Installation")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"[INFO] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("[ERROR] Python 3.8+ required")
        sys.exit(1)
    
    # Install requirements
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"[ERROR] {requirements_file} not found")
        sys.exit(1)
    
    print(f"[INFO] Installing from {requirements_file}")
    
    # Key compatible versions for JAX ecosystem
    critical_packages = [
        "dm-haiku==0.0.14",
        "jax==0.6.2", 
        "jaxlib==0.6.2",
        "optax==0.2.4"
    ]
    
    # Install critical packages first
    for package in critical_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"[ERROR] Failed to install critical package: {package}")
            sys.exit(1)
    
    # Install remaining requirements
    if not run_command(f"pip install -r {requirements_file}", "Installing remaining requirements"):
        print("[WARNING] Some packages may have failed to install")
    
    # Test imports
    print("\n[INFO] Testing critical imports...")
    
    test_imports = [
        ("import jax", "JAX"),
        ("import jax.numpy as jnp", "JAX NumPy"),
        ("import haiku as hk", "Haiku"),
        ("import optax", "Optax"),
        ("from rtdlm import RT_DLM_AGI", "RT-DLM AGI Core")
    ]
    
    all_passed = True
    for import_statement, description in test_imports:
        try:
            exec(import_statement)
            print(f"[PASS] {description}")
        except Exception as e:
            print(f"[FAIL] {description}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] All dependencies installed and tested successfully!")
        print("\nNext steps:")
        print("1. Run system validation: python system_validator.py")
        print("2. Run system demo: python system_demo.py")
        print("3. Execute tests: cd tests && python test_framework.py")
    else:
        print("[WARNING] Some imports failed. Check the errors above.")
        print("You may need to manually resolve dependency conflicts.")

if __name__ == "__main__":
    main()
