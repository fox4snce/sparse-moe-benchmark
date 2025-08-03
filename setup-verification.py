#!/usr/bin/env python3
"""
Verification script for MoE Reality Check setup
"""

import os
import sys
import json
from pathlib import Path


def check_structure():
    """Check that all required directories and files exist"""
    print("🔍 Checking project structure...")
    
    required_dirs = [
        "00_sanity",
        "10_core", 
        "10_core/configs",
        "20_extended",
        "benchmarks",
        "docs"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "Makefile",
        "make.bat",
        "00_sanity/test_router_shapes.py",
        "10_core/run.py",
        "10_core/compare.py",
        "10_core/configs/moe_3expert_quick.yaml",
        "10_core/configs/dense120_quick.yaml",
        "docs/tech_note.md"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            all_good = False
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            all_good = False
        else:
            print(f"✅ File exists: {file_path}")
    
    return all_good


def check_results():
    """Check that benchmark results exist and are valid"""
    print("\n🔍 Checking benchmark results...")
    
    benchmark_dir = Path("benchmarks")
    result_files = list(benchmark_dir.glob("*_results.json"))
    
    if not result_files:
        print("❌ No benchmark results found")
        return False
    
    all_good = True
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            required_fields = ['model', 'tokens_per_sec', 'vram_peak_gb', 'active_params_m']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing field '{field}' in {result_file}")
                    all_good = False
                    break
            else:
                print(f"✅ Valid results: {result_file.name}")
                
        except Exception as e:
            print(f"❌ Error reading {result_file}: {e}")
            all_good = False
    
    return all_good


def check_imports():
    """Check that all required packages can be imported"""
    print("\n🔍 Checking package imports...")
    
    required_packages = [
        'torch',
        'transformers', 
        'numpy',
        'yaml',
        'pytest'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            all_good = False
    
    return all_good


def main():
    """Run all verification checks"""
    print("🚀 MOE REALITY CHECK - SETUP VERIFICATION")
    print("=" * 50)
    
    structure_ok = check_structure()
    results_ok = check_results()
    imports_ok = check_imports()
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if structure_ok and results_ok and imports_ok:
        print("✅ ALL CHECKS PASSED - Setup is ready!")
        print("\nNext steps:")
        print("1. Run: python 10_core/compare.py")
        print("2. Run: .\\make.bat test (Windows) or make test (Linux/Mac)")
        print("3. Ready for publication!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 