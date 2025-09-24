#!/usr/bin/env python3
"""
Test runner for PeppaSync LangChain Application
Run all tests in sequence
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all test suites"""
    print("🧪 PeppaSync LangChain Test Suite")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "tests/tests_api_endpoints.py",
        "tests/tests_advanced_ai.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        print(f"\n🔄 Running {test_file}...")
        try:
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ {test_file} - FAILED")
                print(f"Error: {result.stderr}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {test_file} - ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()