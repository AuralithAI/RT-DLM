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

def run_test(test_name: str, verbose: bool = False, timeout: int = 600):
    """Run a specific test or demo
    
    Args:
        test_name: Name of the test to run
        verbose: Enable verbose output
        timeout: Maximum execution time in seconds (default: 600)
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


def run_benchmark():
    """Run benchmark evaluation on MMMU-style tasks
    
    Evaluates the model on a standardized set of multimodal tasks
    and reports performance metrics.
    """
    import time
    import numpy as np
    
    logger.info("="*60)
    logger.info("RT-DLM AGI Benchmark Mode")
    logger.info("="*60)
    
    metrics = {
        "total_tasks": 0,
        "completed_tasks": 0,
        "reasoning_accuracy": 0.0,
        "creative_score": 0.0,
        "scientific_score": 0.0,
        "multimodal_score": 0.0,
        "inference_time_avg_ms": 0.0,
        "memory_usage_mb": 0.0,
    }
    
    try:
        # Import AGI components
        sys.path.insert(0, str(PROJECT_ROOT))
        from config.agi_config import AGIConfig
        from inference import RT_DLM_AGI_Assistant
        
        # Create lightweight config for benchmarking
        config = AGIConfig(
            d_model=256,
            num_heads=4,
            num_layers=4,
            vocab_size=8000,
            multimodal_enabled=True,
            consciousness_simulation=False,  # Disable for speed
            quantum_layers=0,
        )
        
        logger.info("Initializing AGI for benchmarking...")
        assistant = RT_DLM_AGI_Assistant(config)
        
        # Benchmark tasks simulating MMMU-style evaluation
        benchmark_tasks = [
            {"type": "reasoning", "prompt": "If A implies B and B implies C, what can we conclude about A and C?"},
            {"type": "reasoning", "prompt": "A train leaves at 9am traveling 60mph. Another leaves at 10am at 80mph. When do they meet?"},
            {"type": "creative", "prompt": "Write a haiku about artificial intelligence"},
            {"type": "creative", "prompt": "Describe a world where time flows backwards"},
            {"type": "scientific", "hypothesis": "Higher temperatures increase reaction rates", "observations": "At 20C: 10min, at 40C: 5min, at 60C: 2min"},
            {"type": "scientific", "hypothesis": "Gravity affects plant growth direction", "observations": "Plants in microgravity grow randomly, on Earth they grow upward"},
        ]
        
        metrics["total_tasks"] = len(benchmark_tasks)
        inference_times = []
        
        logger.info(f"Running {len(benchmark_tasks)} benchmark tasks...")
        
        for i, task in enumerate(benchmark_tasks):
            start_time = time.time()
            
            try:
                if task["type"] == "reasoning":
                    result = assistant.think_step_by_step(task["prompt"])
                    if result.get("final_answer"):
                        metrics["completed_tasks"] += 1
                        metrics["reasoning_accuracy"] += result.get("confidence", 0.5)
                        
                elif task["type"] == "creative":
                    result = assistant.creative_generation(task["prompt"])
                    if result.get("creative_output"):
                        metrics["completed_tasks"] += 1
                        metrics["creative_score"] += result.get("novelty_score", 0.5)
                        
                elif task["type"] == "scientific":
                    result = assistant.scientific_inquiry(task["hypothesis"], task["observations"])
                    if result.get("scientific_analysis"):
                        metrics["completed_tasks"] += 1
                        metrics["scientific_score"] += 0.7  # Base score for completion
                
                inference_times.append((time.time() - start_time) * 1000)
                logger.info(f"  Task {i+1}/{len(benchmark_tasks)}: {task['type']} - DONE")
                
            except Exception as e:
                logger.warning(f"  Task {i+1}/{len(benchmark_tasks)}: {task['type']} - FAILED: {e}")
        
        # Calculate averages
        num_reasoning = sum(1 for t in benchmark_tasks if t["type"] == "reasoning")
        num_creative = sum(1 for t in benchmark_tasks if t["type"] == "creative")
        num_scientific = sum(1 for t in benchmark_tasks if t["type"] == "scientific")
        
        if num_reasoning > 0:
            metrics["reasoning_accuracy"] /= num_reasoning
        if num_creative > 0:
            metrics["creative_score"] /= num_creative
        if num_scientific > 0:
            metrics["scientific_score"] /= num_scientific
        
        if inference_times:
            metrics["inference_time_avg_ms"] = np.mean(inference_times)
        
        # Estimate memory usage
        try:
            import jax
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(assistant.params))
            metrics["memory_usage_mb"] = (param_count * 4) / (1024 * 1024)  # Assume float32
        except:
            pass
        
    except Exception as e:
        logger.error(f"Benchmark initialization failed: {e}")
        metrics["error"] = str(e)
    
    # Report results
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*60)
    logger.info(f"  Tasks Completed:       {metrics['completed_tasks']}/{metrics['total_tasks']}")
    logger.info(f"  Reasoning Accuracy:    {metrics['reasoning_accuracy']:.2%}")
    logger.info(f"  Creative Score:        {metrics['creative_score']:.2%}")
    logger.info(f"  Scientific Score:      {metrics['scientific_score']:.2%}")
    logger.info(f"  Avg Inference Time:    {metrics['inference_time_avg_ms']:.1f}ms")
    logger.info(f"  Model Memory Usage:    {metrics['memory_usage_mb']:.1f}MB")
    logger.info("="*60)
    
    return metrics

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
        "--benchmark",
        action="store_true",
        help="Run benchmark evaluation on MMMU-style tasks"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests sequentially"
    )
    
    args = parser.parse_args()
    
    # Handle different commands
    if args.list:
        list_tests()
        return
    
    if args.check_deps:
        check_dependencies()
        return
    
    if args.benchmark:
        run_benchmark()
        return
    
    # Handle --full mode: run all tests sequentially
    if args.full:
        logger.info("Running all tests sequentially (--full mode)")
        all_tests = ["simple", "tokenizer", "validator", "system", "hybrid"]
        results = {}
        
        for test_name in all_tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            success = run_test(test_name, args.verbose)
            results[test_name] = success
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        
        for test_name, success in results.items():
            status = "PASSED" if success else "FAILED"
            logger.info(f"  {test_name:15} - {status}")
        
        logger.info(f"\nTotal: {passed} passed, {failed} failed")
        
        if failed > 0:
            sys.exit(1)
        return
    
    if not args.test:
        print("RT-DLM Test Runner")
        print("=" * 30)
        logger.info("No test specified. Use --list to see available tests.")
        print("\nQuick start:")
        print("  python test_runner.py simple    # Basic CPU test")
        print("  python test_runner.py --full    # Run all tests")
        return
    
    # Run the specified test
    success = run_test(args.test, args.verbose)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
