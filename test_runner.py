#!/usr/bin/env python3
"""
RT-DLM Test Runner
Main entry point for running various demonstrations and tests.
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory using absolute path
PROJECT_ROOT = Path(__file__).parent.resolve()

def run_test(test_name: str, verbose: bool = False, timeout: int = 300):
    """Run a specific test or demo
    
    Args:
        test_name: Name of the test to run
        verbose: Enable verbose output
        timeout: Maximum execution time in seconds (default: 300)
    """
    
    # Define available tests with absolute paths
    tests = {
        "simple": {
            "file": PROJECT_ROOT / "tests" / "demo" / "simple_cpu_demo.py",
            "description": "Basic CPU-only component test",
            "requirements": ["jax", "haiku"]
        },
        "hybrid": {
            "file": PROJECT_ROOT / "tests" / "demo" / "hybrid_demo.py", 
            "description": "Full hybrid AGI system demo (requires GPU/TPU)",
            "requirements": ["jax", "haiku", "requests", "beautifulsoup4"]
        },
        "system": {
            "file": PROJECT_ROOT / "tests" / "demo" / "system_demo.py",
            "description": "Production system demonstration",
            "requirements": ["jax", "haiku"]
        },
        "validator": {
            "file": PROJECT_ROOT / "tests" / "system_validator.py",
            "description": "System validation and readiness check",
            "requirements": ["jax", "haiku"]
        },
        "tokenizer": {
            "file": PROJECT_ROOT / "tests" / "test_tokenizer.py",
            "description": "Multi-modal tokenizer test",
            "requirements": ["sentencepiece"]
        }
    }
    
    if test_name not in tests:
        logger.error(f"Test '{test_name}' not found.")
        logger.info("Available tests:")
        for name, info in tests.items():
            logger.info(f"  - {name}: {info['description']}")
        return False
    
    test_info = tests[test_name]
    test_file = test_info["file"]
    
    # Check if file exists
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return False
    
    logger.info(f"Running test: {test_name}")
    logger.info(f"Description: {test_info['description']}")
    logger.info(f"File: {test_file}")
    
    # Check requirements
    logger.info(f"Required packages: {', '.join(test_info['requirements'])}")
    
    try:
        # Run the test with timeout
        cmd = [sys.executable, str(test_file)]
        if verbose:
            logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=False, 
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT
        )
        
        if result.returncode == 0:
            logger.info(f"Test '{test_name}' completed successfully!")
            return True
        else:
            logger.error(f"Test '{test_name}' failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"Test '{test_name}' timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"Failed to run test '{test_name}': {e}")
        return False

def list_tests():
    """List all available tests"""
    logger.info("Available RT-DLM Tests:")
    print("=" * 50)
    
    tests = {
        "simple": "Basic CPU-only component test - lightweight, no GPU needed",
        "hybrid": "Full hybrid AGI system demo - requires GPU/TPU for best performance", 
        "system": "Production system demonstration - core AGI capabilities showcase",
        "validator": "System validation and readiness check - production deployment check",
        "tokenizer": "Multi-modal tokenizer test - tests data processing capabilities"
    }
    
    for name, description in tests.items():
        print(f"  {name:12} - {description}")
    
    print("\nUsage:")
    print("  python test_runner.py simple      # Run basic CPU test")
    print("  python test_runner.py hybrid      # Run full hybrid demo")
    print("  python test_runner.py validator   # Run system validation")
    print("  python test_runner.py --list      # Show this list")

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = {
        "jax": "JAX numerical computing library",
        "haiku": "Haiku neural network library", 
        "numpy": "NumPy numerical arrays",
        "optax": "Optax optimization library"
    }
    
    optional_packages = {
        "requests": "HTTP requests (for web integration)",
        "bs4": "HTML parsing (for web scraping)",
        "sentencepiece": "SentencePiece tokenizer"
    }
    
    print("\nRequired packages:")
    all_required_found = True
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {package:15} - {description}")
        except ImportError:
            print(f"  [MISSING] {package:15} - {description}")
            all_required_found = False
    
    print("\nOptional packages:")
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {package:15} - {description}")
        except ImportError:
            print(f"  [MISSING] {package:15} - {description}")
    
    return all_required_found

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RT-DLM Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py simple          # Run basic CPU test
  python test_runner.py hybrid --verbose # Run hybrid demo with verbose output
  python test_runner.py --check-deps    # Check dependencies
  python test_runner.py --list          # List available tests
        """
    )
    
    parser.add_argument(
        "test", 
        nargs="?", 
        help="Test to run (simple, hybrid, tokenizer)"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available tests"
    )
    
    parser.add_argument(
        "--check-deps", 
        action="store_true", 
        help="Check dependencies"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.list:
        list_tests()
        return
    
    if args.check_deps:
        check_dependencies()
        return
    
    if not args.test:
        print("RT-DLM Test Runner")
        print("=" * 30)
        logger.info("No test specified. Use --list to see available tests.")
        print("\nQuick start:")
        print("  python test_runner.py simple    # Basic CPU test")
        return
    
    # Run the specified test
    success = run_test(args.test, args.verbose)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
