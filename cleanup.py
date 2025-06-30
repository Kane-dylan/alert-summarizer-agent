#!/usr/bin/env python3
"""
Simple cleanup script for Alert Summarizer Agent
Removes unnecessary files and optimizes the codebase
"""

import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()

def clean_python_cache():
    """Remove all __pycache__ directories"""
    print("🧹 Cleaning Python cache files...")
    count = 0
    for pycache_dir in ROOT_DIR.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            count += 1
            print(f"  Removed: {pycache_dir}")
        except Exception as e:
            print(f"  Failed to remove {pycache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in ROOT_DIR.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            count += 1
            print(f"  Removed: {pyc_file}")
        except Exception as e:
            print(f"  Failed to remove {pyc_file}: {e}")
    
    print(f"✅ Removed {count} cache files")

def clean_model_cache():
    """Clean model cache directories"""
    print("🧹 Cleaning model cache...")
    
    # Local cache
    cache_dirs = [
        ROOT_DIR / ".cache",
        ROOT_DIR / ".model_cache",
        ROOT_DIR / "models"
    ]
    
    count = 0
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                count += 1
                print(f"  Removed: {cache_dir}")
            except Exception as e:
                print(f"  Failed to remove {cache_dir}: {e}")
    
    print(f"✅ Removed {count} model cache directories")

def clean_virtual_env():
    """Remove virtual environment if it exists"""
    print("🧹 Cleaning virtual environment...")
    venv_dir = ROOT_DIR / "venv"
    if venv_dir.exists():
        try:
            shutil.rmtree(venv_dir)
            print(f"  Removed: {venv_dir}")
            print("✅ Virtual environment removed")
        except Exception as e:
            print(f"  Failed to remove {venv_dir}: {e}")
    else:
        print("  No virtual environment found")

def show_project_size():
    """Show project size information"""
    print("📊 Project size analysis...")
    
    total_size = 0
    file_count = 0
    
    for file in ROOT_DIR.rglob("*"):
        if file.is_file() and not file.match(".git/*"):
            try:
                size = file.stat().st_size
                total_size += size
                file_count += 1
            except:
                pass
    
    # Convert to MB
    total_mb = total_size / (1024 * 1024)
    
    print(f"  Total files (excluding .git): {file_count}")
    print(f"  Total size: {total_mb:.2f} MB")

def main():
    """Main cleanup function"""
    print("🧹 Alert Summarizer Agent - Cleanup")
    print("=" * 40)
    
    # Show current state
    show_project_size()
    print()
    
    # Cleanup operations
    clean_python_cache()
    clean_model_cache()
    clean_virtual_env()
    
    print()
    print("🎉 Cleanup completed!")
    print()
    
    # Show final state
    show_project_size()
    
    print()
    print("📋 Next steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Unix)")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run tests: python test_runner.py")
    print("5. Start app: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
