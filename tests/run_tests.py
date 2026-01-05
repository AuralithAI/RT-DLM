"""
Test runner for RT-DLM AGI components
"""

import sys
import os
import subprocess
from pathlib import Path


def run_single_test(test_path: Path, project_root: Path) -> str:
    """Run a single test file and return result status.
    
    Args:
        test_path: Path to the test file
        project_root: Path to the project root directory
        
    Returns:
        'pass', 'fail', 'timeout', or 'error'
    """
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            return 'pass'
        else:
            # Show only last few lines of error
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')[-5:]
                print(f"Error: {os.linesep.join(error_lines)}")
            return 'fail'
            
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception as e:
        print(f"Exception: {e}")
        return 'error'


def run_tests():
    """Run all tests in the tests directory."""
    
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    
    print("[TEST] Running RT-DLM AGI Tests")
    print("=" * 40)
    
    # Test files to run - using os.sep for cross-platform compatibility
    test_files = [
        # Core AGI module tests (new split tests)
        "test_consciousness.py",
        "test_scientific.py",
        "test_creative.py",
        "test_emotional.py",
        "test_audio_emotion.py",
        "test_integration.py",
        "test_training.py",
        # Hybrid fusion tests
        "test_hybrid_fusion.py",
        # Tokenizer test
        "test_tokenizer.py",
        # Example tests
        os.sep.join(["test_example", "test_attention.py"]),
        os.sep.join(["test_example", "test_embedding.py"]),
        os.sep.join(["test_example", "test_moe.py"]),
        os.sep.join(["test_example", "test_transformer_block.py"]),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        
        if not test_path.exists():
            print(f"[SKIP] {test_file} - NOT FOUND")
            skipped += 1
            continue
            
        print(f"\n[RUN] Running {test_file}...")
        status = run_single_test(test_path, project_root)
        
        if status == 'pass':
            print(f"[PASS] {test_file} - PASSED")
            passed += 1
        elif status == 'timeout':
            print(f"[TIMEOUT] {test_file} - TIMEOUT")
            failed += 1
        else:
            print(f"[FAIL] {test_file} - FAILED")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"[RESULTS] Test Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("[OK] All tests passed! RT-DLM AGI is ready!")
    elif failed > 0:
        print("[WARN] Some tests failed. Check the output above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
