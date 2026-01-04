"""
Test runner for RT-DLM AGI components
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory."""
    
    tests_dir = Path(__file__).parent
    
    print("[TEST] Running RT-DLM AGI Tests")
    print("=" * 40)
    
    # Test files to run
    test_files = [
        "test_tokenizer.py",
        "test_example/test_attention.py",
        "test_example/test_embedding.py", 
        "test_example/test_moe.py",
        "test_example/test_transformer_block.py"
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = tests_dir / test_file
        
        if test_path.exists():
            print(f"\n[RUN] Running {test_file}...")
            try:
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    cwd=tests_dir.parent,  # Run from project root
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    print(f"[PASS] {test_file} - PASSED")
                    passed += 1
                else:
                    print(f"[FAIL] {test_file} - FAILED")
                    print(f"Error: {result.stderr}")
                    failed += 1
                    
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {test_file} - TIMEOUT")
                failed += 1
            except Exception as e:
                print(f"[ERROR] {test_file} - ERROR: {e}")
                failed += 1
        else:
            print(f"[WARN] {test_file} - NOT FOUND")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"[RESULTS] Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("[OK] All tests passed! RT-DLM AGI is ready!")
    else:
        print("[WARN] Some tests failed. Check the output above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
